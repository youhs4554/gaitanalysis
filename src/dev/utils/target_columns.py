def get_target_columns(opt):
    # spatial params
    spatial_params = [
        'Stride Length(cm)/L', 'Stride Length(cm)/R',
        'HH Base Support(cm)/L', 'HH Base Support(cm)/R',
    ]

    # temporal params
    temporal_params = [
        'Cycle Time(sec)/L', 'Cycle Time(sec)/R',
        'Stance Time(sec)/L', 'Stance Time(sec)/R',
        'Double Supp. Time(sec)/L', 'Double Supp. Time(sec)/R',
        'Swing Time(sec)/L', 'Swing Time(sec)/R',
    ]

    # etc params
    etc_params = [
        'Functional Amb. Profile',
        'Velocity',
        'Cadence',
    ]

    global group_map

    group_map = {
        'all': spatial_params + temporal_params + etc_params,
        'spatial': spatial_params,
        'temporal': temporal_params,
        'etc': etc_params
    }

    assert opt.target_columns in group_map.keys(), "Invalid columns names for target"

    return group_map[opt.target_columns]


def get_target_columns_by_group(group):
    return group_map[group]
