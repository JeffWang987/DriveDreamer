def format_size(x):
    if x > 1e8:
        return '{:.1f}G'.format(x / 1e9)
    if x > 1e5:
        return '{:.1f}M'.format(x / 1e6)
    if x > 1e2:
        return '{:.1f}K'.format(x / 1e3)
    return str(x)


def parameter_count(model):
    count = 0
    for parameter in model.parameters():
        count += parameter.numel()
    return count, format_size(count)
