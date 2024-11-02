import random

box_label_table = {
    'non_vehicle': 'non_motorized_vehicle',
    'tricycle': 'freight_tricycle',
    'tri_reflector': 'roadside_warning_triangle',
}


box_label_fix_keys = ['vehicle', 'non_vehicle']

box_label_probs = {}


def remove_none_label(labels):
    if not isinstance(labels, (list, tuple)):
        labels = [labels]
    new_labels = []
    for label in labels:
        if label is not None:
            new_labels.append(label)
    return new_labels


def generate_box_label(labels, random_choice=False):
    labels = remove_none_label(labels)
    if len(labels) == 0:
        return None
    if random_choice:
        box_labels = [labels[0]]
        if labels[0] in box_label_fix_keys and len(labels) > 1:
            box_labels.append(labels[1])
            labels = labels[2:]
        else:
            labels = labels[1:]
        for label in labels:
            prob = box_label_probs.get(label, 0.5)
            if random.random() < prob:
                box_labels.append(label)
        box_label = ','.join(box_labels)
    else:
        box_label = ','.join(labels)
    return box_label
