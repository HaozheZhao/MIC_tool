
import re
import copy as cp
from pathlib import Path
import random
import common.io_utils as ioutils


def gather_templates(template_path):
    """
        Gather all templates under template_path
    """
    tmpl_path = Path(template_path)
    prompt_pattern = "\{[a-zA-Z0-9\_\-]*\}"
    all_templates = []
    all_params = set()
    output_templates = []
    output_params = set()
    prompt_templs = ioutils.load_json(tmpl_path)
    if isinstance(prompt_templs["template"],list):
        for tmpl in prompt_templs["template"]:
            all_templates.append({"template": tmpl, "source": prompt_templs["source"]})
            all_params.update([x[1 : -1] for x in re.findall(prompt_pattern, tmpl)])
    else:
        input_template = prompt_templs["template"]['input_template']
        output_template = prompt_templs["template"]['output_template']
        for tmpl in input_template:
            all_templates.append({"template": tmpl, "source": prompt_templs["source"]})
            all_params.update([x[1 : -1] for x in re.findall(prompt_pattern, tmpl)])
        for tmpl in output_template:
            output_templates.append({"template": tmpl, "source": prompt_templs["source"]})
            output_params.update([x[1 : -1] for x in re.findall(prompt_pattern, tmpl)])
        return [{"template": all_templates, "param": all_params},{"template": output_templates, "param": output_params}]
    
    return {"template": all_templates, "param": all_params}


def initialize_template(prompt_templates, vocabulary, recursive=False):
    """
        Initialize template with all substitutable params in dataset
    """

    # Recursively generate all possible initialization of a template
    def dfs(template, params, vocabulary):
        results = []
        substituted = False
        for param in params:
            if param in template:
                substituted = True
                assert isinstance(vocabulary[param], list), "for recursive initialization, " \
                                                            "vocabulary should contain multiple options for each param."
                for option in vocabulary[param]:
                    new_template = cp.deepcopy(template)
                    new_template.replace("{" + param + "}", option)
                    results += dfs(new_template, params, vocabulary)
        if not substituted:
            return [template]
        return results

    if isinstance(prompt_templates,dict):
        templates = prompt_templates["template"]
        params = prompt_templates["param"]

        prompt_template = random.choice(templates)
        # for prompt_template in templates:
        if recursive:
            prompts = dfs(prompt_template["template"], params, vocabulary)
        else:
            # Generate taskgen for only one instance, substitute parameters accordingly
            prompt = cp.deepcopy(prompt_template["template"])
            for param in params:
                if param in vocabulary.keys():
                    prompt = prompt.replace("{" + param + "}", vocabulary[param])
                    prompt = prompt.replace("..",'.')
            prompts = [prompt]

        all_prompts = [{"prompt": prompt, "source": prompt_template["source"]} for prompt in prompts]
    else:
        all_prompts = []
        for p_template in prompt_templates:
            templates = p_template["template"]
            params = p_template["param"]

            prompt_template = random.choice(templates)
            # for prompt_template in templates:
            if recursive:
                prompts = dfs(prompt_template["template"], params, vocabulary)
            else:
                # Generate taskgen for only one instance, substitute parameters accordingly
                prompt = cp.deepcopy(prompt_template["template"])
                for param in params:
                    if param in vocabulary.keys():
                        prompt = prompt.replace("{" + param + "}", vocabulary[param])
                        prompt = prompt.replace("..",'.')
                prompts = [prompt]

            all_prompts += [{"prompt": prompt, "source": prompt_template["source"]} for prompt in prompts]
    # Notice: treating same taskgen templates from different sources as different prompts, could change all_prompts and
    #         stringify of taskgen to change it to unique ones.
    return all_prompts