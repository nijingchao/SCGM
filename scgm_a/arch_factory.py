from torchvision.models import resnet18, resnet34, resnet50
from resnets import resnet12

model_pool = {
    'resnet12': lambda num_classes=2: resnet12(avg_pool=True, drop_rate=0.1, dropblock_size=5, num_classes=num_classes),
    'resnet12forcifar': lambda num_classes=2: resnet12(avg_pool=True, drop_rate=0.1, dropblock_size=2, num_classes=num_classes),
    'resnet18': resnet18,
    'resnet34': resnet34,
    'resnet50': resnet50
}


class ArchFactory(object):
    def create_arch(self, arch_name):
        encoder_q, encoder_k = model_pool[arch_name](), model_pool[arch_name]()
        k2q_mapping = {k_name: q_name for q_name, k_name in zip(encoder_q.state_dict().keys(), encoder_k.state_dict().keys())}

        return encoder_q, encoder_k, k2q_mapping
