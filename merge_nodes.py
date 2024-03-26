import pathlib
import re
import time
from typing import List, Literal
import sd_mecha
import torch.cuda
from invokeai.app.services.shared.invocation_context import InvocationContext
from sd_mecha.extensions.merge_method import MergeMethod, convert_to_recipe
import pydantic
from invokeai import invocation_api


VERSION = "0.0.1"


class RecipeField(pydantic.BaseModel):
    """
    Mecha recipe serialized
    """
    serialized_recipe: str = pydantic.Field(description="Serialized mecha recipe")


ALL_CUDA_DEVICES = ["cpu", *(["cuda", *[f"cuda:{i}" for i in range(torch.cuda.device_count())]] if torch.cuda.is_available() else [])]
DTYPE_MAPPING = {
    "bf16": torch.bfloat16,
    "fp16": torch.float16,
    "fp32": torch.float32,
    "fp64": torch.float64,
}
OPTIONAL_DTYPE_MAPPING = {
    "default": None,
} | DTYPE_MAPPING


@invocation_api.invocation("mecha_recipe_merger", "Recipe Merger", version=VERSION)
class RecipeMergerInvocation(invocation_api.BaseInvocation):
    recipe: RecipeField = invocation_api.InputField(description="Recipe to merge")
    fallback_model: invocation_api.ModelIdentifierField = invocation_api.InputField(
        description="Fallback model for the VAE and missing keys",
        input=invocation_api.Input.Direct,
        ui_type=invocation_api.UIType.MainModel,
    )
    output_name: str = invocation_api.InputField(
        description="Name of the model to save under the models dir",
        default="mecha-merge",
    )
    merge_device: Literal[*ALL_CUDA_DEVICES] = invocation_api.InputField(
        description="Device used to merge",
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    merge_dtype: Literal[*DTYPE_MAPPING.keys()] = invocation_api.InputField(
        description="Precision used to merge",
        default="fp64",
    )
    output_dtype: Literal[*DTYPE_MAPPING.keys()] = invocation_api.InputField(
        description="Precision used to store the merged model",
        default="fp16",
    )
    total_buffer_size: str = invocation_api.InputField(
        description="Maximum total memory that can be used to store input models (G=gigabyte, M=megabyte, K=kilobyte, nothing=byte)",
        default="1G",
    )
    threads: int = invocation_api.InputField(
        description="Number of threads to merge the model (0=automatic)",
        ge=0,
        le=16,
        default=0,
    )

    def invoke(self, context: InvocationContext) -> invocation_api.ModelLoaderOutput:
        recipe = sd_mecha.deserialize(self.recipe.serialized_recipe.split("\n"))
        model_arch = getattr(recipe.model_arch, "identifier", None)
        if self.fallback_model is None or not model_arch:
            fallback_model = None
        else:
            fallback_model = context._services.model_manager.store.get_model(self.fallback_model.key).path
            fallback_model = sd_mecha.model(fallback_model, model_arch=model_arch)

        merger = sd_mecha.RecipeMerger(
            default_device=self.merge_device,
            default_dtype=DTYPE_MAPPING[self.merge_dtype],
        )
        model_path = (pathlib.Path(context.config.get().models_path) / self.output_name).with_suffix(".safetensors")

        merged_configs = context.models.search_by_path(model_path)
        if merged_configs:
            context._services.model_manager.install.unregister(merged_configs[0].key)
            context._services.model_manager.install.wait_for_installs()

        merger.merge_and_save(
            recipe=recipe,
            output=model_path,
            fallback_model=fallback_model,
            save_dtype=DTYPE_MAPPING[self.output_dtype],
            save_device="cpu",
            threads=self.threads if self.threads > 0 else None,
            total_buffer_size=parse_memory(self.total_buffer_size),
        )
        key = context._services.model_manager.install.install_path(str(model_path))
        merged_model = invocation_api.ModelIdentifierField.from_config(context.models.get_config(key))

        unet = merged_model.model_copy(update={"submodel_type": invocation_api.SubModelType.UNet})
        scheduler = merged_model.model_copy(update={"submodel_type": invocation_api.SubModelType.Scheduler})
        tokenizer = merged_model.model_copy(update={"submodel_type": invocation_api.SubModelType.Tokenizer})
        text_encoder = merged_model.model_copy(update={"submodel_type": invocation_api.SubModelType.TextEncoder})
        vae = merged_model.model_copy(update={"submodel_type": invocation_api.SubModelType.VAE})

        return invocation_api.ModelLoaderOutput(
            unet=invocation_api.UNetField(unet=unet, scheduler=scheduler, loras=[]),
            clip=invocation_api.CLIPField(tokenizer=tokenizer, text_encoder=text_encoder, loras=[], skipped_layers=0),
            vae=invocation_api.VAEField(vae=vae),
        )


def parse_memory(size_expr: str) -> int:
    total = 0
    size_expr = size_expr.replace(" ", "")
    split = re.split(r"(\.[0-9]+|[0-9]+(?:\.[0-9]*)?)([a-zA-Z]?)", size_expr)
    for amount, modifier in zip(split[1::3], split[2::3]):
        amount = float(amount)
        if modifier:
            amount *= {
                "k": 1024,
                "m": 1024**2,
                "g": 1024**3,
            }.get(modifier, 1)
        total += amount

    return min(max(2**8, total), 2**32)


@invocation_api.invocation_output('recipe_invocation_output')
class RecipeInvocationOutput(invocation_api.BaseInvocationOutput):
    """
    Mecha Recipe Output
    """

    recipe: RecipeField = invocation_api.OutputField(description="Mecha recipe")


@invocation_api.invocation("mecha_model_recipe", "Model Recipe", version=VERSION)
class ModelRecipeInvocation(invocation_api.BaseInvocation):
    """
    Mecha Model Recipe
    """

    model: invocation_api.ModelIdentifierField = invocation_api.InputField(
        input=invocation_api.Input.Direct,
        ui_type=invocation_api.UIType.MainModel,
        description=invocation_api.FieldDescriptions.main_model,
    )

    def invoke(self, context: invocation_api.InvocationContext) -> RecipeInvocationOutput:
        model_arch = {
            invocation_api.BaseModelType.StableDiffusion1: "sd1",
            invocation_api.BaseModelType.StableDiffusionXL: "sdxl",
        }[self.model.base]
        model_type = {
            invocation_api.ModelType.Main: "base",
            invocation_api.ModelType.LoRA: "lora",
        }[self.model.type]
        model_path = context._services.model_manager.store.get_model(self.model.key).path
        return RecipeInvocationOutput(
            recipe=RecipeField(serialized_recipe=sd_mecha.serialize(sd_mecha.model(model_path, model_arch, model_type)))
        )


def register_merge_methods():
    for method_name in sd_mecha.extensions.merge_method._merge_methods_registry:
        method = sd_mecha.extensions.merge_method.resolve(method_name)
        sub_snake_name = f"{method_name}_recipe"
        invocation_name = f"mecha_{sub_snake_name}"
        title_name = snake_case_to_title(sub_snake_name)
        class_name = f"{snake_case_to_upper(sub_snake_name)}Invocation"
        register_merge_method_invoke_node(class_name, invocation_name, title_name, method)


def register_merge_method_invoke_node(class_name: str, invocation_name: str, title: str, method: MergeMethod):
    cls = type(class_name, (invocation_api.BaseInvocation,), {
        "__doc__": title,
        "__annotations__": {
            **{
                model_name: RecipeField
                for model_name, merge_space in zip(method.get_model_names(), method.get_input_merge_spaces()[0])
            },
            **({method.get_model_varargs_name(): List[RecipeField]} if method.get_model_varargs_name() is not None else {}),
            **{
                hyper_name: float
                for hyper_name in method.get_hyper_names() - method.get_volatile_hyper_names()
            },
        },
        **{
            model_name: invocation_api.InputField(
                description=f"Recipe '{model_name}' ({merge_space})",
            )
            for model_name, merge_space in zip(method.get_model_names(), method.get_input_merge_spaces()[0])
        },
        **(
            {method.get_model_varargs_name(): invocation_api.InputField(description=f"Variadic Recipes ({method.get_input_merge_spaces()[1]})")}
            if method.get_model_varargs_name() is not None
            else {}
        ),
        **{
            hyper_name: invocation_api.InputField(
                default=method.get_default_hypers()[hyper_name],
                description=f"Optional Hyperparameter '{hyper_name}'",
            )
            for hyper_name in method.get_hyper_names() - method.get_volatile_hyper_names()
            if hyper_name in method.get_default_hypers()
        },
        **{
            hyper_name: invocation_api.InputField(
                description=f"Hyperparameter '{hyper_name}'",
            )
            for hyper_name in method.get_hyper_names() - method.get_volatile_hyper_names()
            if hyper_name not in method.get_default_hypers()
        },
        "invoke": get_method_invoke_function(method)
    })
    return invocation_api.invocation(invocation_name, title=title, version=VERSION)(cls)


def get_method_invoke_function(method: MergeMethod):
    def invoke(self, context: invocation_api.InvocationContext) -> RecipeInvocationOutput:
        # dtype = OPTIONAL_DTYPE_MAPPING[self.dtype]
        # device = self.device
        # if device == "default":
        #     device = None

        models = [
            sd_mecha.deserialize(getattr(self, m).serialized_recipe.split("\n"))
            for m in method.get_model_names()
        ] + (
            [sd_mecha.deserialize(m.serialized_recipe.split("\n")) for m in getattr(self, method.get_model_varargs_name())]
            if method.get_model_varargs_name() is not None
            else []
        )
        hypers = {
            k: getattr(self, k)
            for k in method.get_hyper_names()
            if getattr(self, k) is not None
        }

        return RecipeInvocationOutput(
            # recipe=sd_mecha.serialize(method.create_recipe(*models, **hypers, dtype=dtype, device=device))
            recipe=RecipeField(serialized_recipe=sd_mecha.serialize(method.create_recipe(*models, **hypers)))
        )

    return invoke


def snake_case_to_upper(name: str):
    i = 0
    while i < len(name):
        if name[i] == "_":
            name = name[:i] + name[i+1:i+2].upper() + name[i+2:]
        i += 1

    return name[:1].upper() + name[1:]


def snake_case_to_title(name: str):
    i = 0
    while i < len(name):
        if name[i] == "_":
            name = name[:i] + " " + name[i+1:i+2].upper() + name[i+2:]
        i += 1

    return name[:1].upper() + name[1:]



register_merge_methods()
