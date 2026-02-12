from methods.lorasub_drs import LoRAsub_DRS
from methods.sprompt_coda import SPrompts_coda
from methods.sprompt_l2p import SPrompts_l2p
from methods.sprompt_dual import SPrompts_dual


def get_model(model_name, args):
    name = model_name.lower()
    options = {'lorasub_drs': LoRAsub_DRS,
               'sprompts_coda': SPrompts_coda,
               'sprompts_l2p': SPrompts_l2p,
               'sprompts_dual': SPrompts_dual,
               }
    return options[name](args)

