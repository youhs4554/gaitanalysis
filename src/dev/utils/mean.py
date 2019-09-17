def get_mean(norm_value=255, dataset='activitynet'):
    assert dataset in ['imagenet', 'activitynet', 'kinetics']

    if dataset == 'imagenet':
        return [
            123.675 / norm_value, 116.28 / norm_value, 103.53 / norm_value]
    elif dataset == 'activitynet':
        return [
            114.7748 / norm_value, 107.7354 / norm_value, 99.4750 / norm_value
        ]
    elif dataset == 'kinetics':
        # Kinetics (10 videos for each class)
        return [
            110.63666788 / norm_value, 103.16065604 / norm_value,
            96.29023126 / norm_value
        ]


def get_std(norm_value=255, dataset='activitynet'):
    assert dataset in ['imagenet', 'activitynet', 'kinetics']

    if dataset == 'imagenet':
        return [
            58.395 / norm_value, 57.12 / norm_value, 57.375 / norm_value
        ]
    else:
        # Kinetics (10 videos for each class)
        return [
            38.7568578 / norm_value, 37.88248729 / norm_value,
            40.02898126 / norm_value
        ]
