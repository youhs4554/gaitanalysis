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
            0.43216, 0.394666, 0.37645
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
            0.22803, 0.22145, 0.216989
        ]
