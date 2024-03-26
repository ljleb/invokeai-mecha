# import sd_mecha
#
#
# class BlocksMechaHyper:
#     @classmethod
#     def INPUT_TYPES(cls):
#         return {
#             "required": {
#                 "blocks": ("STRING", {
#                     "default": "",
#                 }),
#                 "default": ("FLOAT", {
#                     "default": 0.0,
#                     "min": -2**64,
#                     "max": 2**64,
#                     "step": 0.1,
#                 }),
#                 "model_arch": (sd_mecha.extensions.model_arch.get_all(),),
#                 "model_component": (["unet", "txt", "txt2"], {
#                     "default": "unet",
#                 }),
#                 "validate_num_blocks": ("BOOLEAN", {
#                     "default": True,
#                 }),
#             },
#         }
#     RETURN_TYPES = ("HYPER",)
#     FUNCTION = "execute"
#     OUTPUT_NODE = False
#     CATEGORY = "advanced/model_merging/mecha"
#
#     def execute(
#         self,
#         blocks: str,
#         default: float,
#         model_arch: str,
#         model_component: str,
#         validate_num_blocks: bool,
#     ):
#         try:
#             return sd_mecha.default(
#                 model_arch=model_arch,
#                 value=default,
#             ) | sd_mecha.blocks(
#                 model_arch,
#                 model_component if model_component else None,
#                 *((float(block.strip()) for block in blocks.split(",")) if blocks.strip() else ()),
#                 strict=validate_num_blocks,
#             ),
#         except ValueError as e:
#             raise ValueError(f"Wrong number of blocks for model architecture '{model_arch}'") from e
#
#
# NODE_CLASS_MAPPINGS = {
#     "Blocks Mecha Hyper": BlocksMechaHyper,
# }
#
# NODE_DISPLAY_NAME_MAPPINGS = {
#     "Blocks Mecha Hyper": "Blocks",
# }
