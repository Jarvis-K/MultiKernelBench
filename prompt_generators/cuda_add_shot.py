from prompt_generators.prompt_registry import register_prompt, BasePromptStrategy
from prompt_generators.prompt_utils import generate_template, read_relavant_files

@register_prompt("cuda", "add_shot")
class CudaDefaultPromptStrategy(BasePromptStrategy):
    def generate(self, op) -> str:
        arc_src, example_arch_src, example_new_arch_src = read_relavant_files('cuda', op, 'add')
        return generate_template(arc_src, example_arch_src, example_new_arch_src, 'CUDA')
