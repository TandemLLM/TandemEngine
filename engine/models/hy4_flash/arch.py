"""The hy_v4_flash family's entry in the model registry (engine/models/__init__.py discovers it)."""

from engine.models import Arch, register
from engine.models.adapter import EffortSpec, ReasoningSpec, ServeSpec

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
    # the arg-key form, which the model writes with suffixed tags (tokenizer.py gives them their plain names);
    # side slots (ENG-485: lanes that take turns, each the full window). Its template takes three modes (max, high,
    # no_think; the pack's chat_template.jinja): the levels from low to high are its high, and only a request's own
    # xhigh or max is its max. With no effort asked and nothing else deciding, medium: the mode high with that
    # level's cap (SRV-899, ENG-468). A block a budget ends closes as the template ends a finished one, its
    # </think:opensource> with the answer straight after, in place of the shared close (Qwen's vendor sentence, then
    # blank lines, which the template never writes). On the raw route the scouting run on 7f2a41f5 found budget-closed
    # answers holding reasoning, the sentence itself once; that the shared close caused it is not shown
    spec=ServeSpec(tool_format="argkey", side_slots=True, reasoning=ReasoningSpec(close="</think>", effort=EffortSpec(
        words=(("low", "high"), ("medium", "high"), ("high", "high"), ("xhigh", "max")),
        default="medium"))),
    # Flash-Next's reference primitives (the stable top-k, the expert bank, an EXL3 projection's product) and
    # its n-gram table readers (the O_DIRECT row reader, the codebook)
    uses=("engine/models/flashnext/ref.py", "engine/models/flashnext/ple.py"),
))
