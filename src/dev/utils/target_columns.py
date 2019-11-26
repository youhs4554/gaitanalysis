def get_target_columns(opt):
    target_columns = ['Velocity', 'Cadence', 'Cycle Time(sec)/L', 'Cycle Time(sec)/R',
                      'Stride Length(cm)/L', 'Stride Length(cm)/R', 'HH Base Support(cm)/L',
                      'HH Base Support(cm)/R', 'Swing Time(sec)/L', 'Swing Time(sec)/R',
                      'Stance Time(sec)/L', 'Stance Time(sec)/R', 'Double Supp. Time(sec)/L',
                      'Double Supp. Time(sec)/R',
                      'Swing % of Cycle/L', 'Swing % of Cycle/R', 'Stance % of Cycle/L',
                      'Stance % of Cycle/R', 'Double Supp % Cycle/L', 'Double Supp % Cycle/R',
                      'Toe In / Out/L', 'Toe In / Out/R',
                      'Stride Length Std Dev/L', 'Stride Length Std Dev/R',
                      'Stride Time Std Dev/L', 'Stride Time Std Dev/R', 'CV Stride Length/L',
                      'CV Stride Length/R', 'CV Stride Time/L', 'CV Stride Time/R']

    opt.target_columns_to_train = target_columns_to_train = \
        ['Velocity', 'Cadence', 'Cycle Time(sec)/L', 'Cycle Time(sec)/R',
         'Stride Length(cm)/L', 'Stride Length(cm)/R', 'HH Base Support(cm)/L',
         'HH Base Support(cm)/R', 'Swing Time(sec)/L', 'Swing Time(sec)/R',
         'Stance Time(sec)/L', 'Stance Time(sec)/R', 'Double Supp. Time(sec)/L',
         'Double Supp. Time(sec)/R', 'Toe In / Out/L', 'Toe In / Out/R',
         'Stride Length Std Dev/L', 'Stride Length Std Dev/R',
         'Stride Time Std Dev/L', 'Stride Time Std Dev/R']

    opt.target_columns_to_eval = target_columns_to_eval = \
        ['Velocity', 'Cadence', 'Cycle Time(sec)/L', 'Cycle Time(sec)/R',
         'Stride Length(cm)/L', 'Stride Length(cm)/R', 'HH Base Support(cm)/L',
         'HH Base Support(cm)/R', 'Swing % of Cycle/L', 'Swing % of Cycle/R',
         'Stance % of Cycle/L', 'Stance % of Cycle/R', 'Double Supp % Cycle/L',
         'Double Supp % Cycle/R', 'Toe In / Out/L', 'Toe In / Out/R',
         'CV Stride Length/L', 'CV Stride Length/R',
         'CV Stride Time/L', 'CV Stride Time/R']

    if opt.model_arch == 'AGNet-pretrain':
        return [
            'Velocity', 'Cadence', 'Cycle Time(sec)/L', 'Cycle Time(sec)/R',
            'Stride Length(cm)/L', 'Stride Length(cm)/R', 'HH Base Support(cm)/L',
            'HH Base Support(cm)/R', 'Swing % of Cycle/L', 'Swing % of Cycle/R',
            'Stance % of Cycle/L', 'Stance % of Cycle/R', 'Double Supp % Cycle/L',
            'Double Supp % Cycle/R', 'Toe In / Out/L', 'Toe In / Out/R'
        ]

    return target_columns_to_eval
