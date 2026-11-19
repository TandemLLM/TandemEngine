"""The hy_v4_flash family's entry in the model registry (engine/models/__init__.py discovers it)."""

from engine.models import Arch, register
from engine.models.adapter import ServeSpec

ARCH = register(Arch(
    name="hy_v4_flash",
    architectures=("HYV4FlashForConditionalGeneration", "HYV4FlashForCausalLM"),
    model_types=("hy_v4_flash", "hy_v4_flash_text"),
    config="engine.models.hy4_flash.config:load_hy4_flash_config",
    engine="engine.models.hy4_flash.engine:Hy4FlashEngine",
    package="models/hy4_flash",
    serve="engine.models.hy4_flash.serving:serve",
    manifest="engine.models.hy4_flash.serving:manifest",
    rank=6,
    # one sequence at a time until the row path has its own kernels and graphs
    spec=ServeSpec(tool_format="argkey", side_slots=False),
    # Flash-Next's reference primitives (the stable top-k, the expert bank, an EXL3 projection's product) and
    # its n-gram table readers (the O_DIRECT row reader, the codebook)
    uses=("engine/models/flashnext/ref.py", "engine/models/flashnext/ple.py"),
))
