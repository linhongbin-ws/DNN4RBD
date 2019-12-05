from regularizeTool import EarlyStopping
from trainTool import train
from loadDataTool import load_train_N_validate_data
from os.path import join
from evaluateTool import *
import scipy.io as sio
from os import mkdir
from loadModel import get_model

def loop_func(train_data_path, valid_data_path, test_data_path, use_net):
    # config hyper-parameters
    max_training_epoch = 2000 # stop train when reach maximum training epoch
    goal_loss = 1e-4 # stop train when reach goal loss
    valid_ratio = 0.2 # ratio of validation data set over train and validate data
    batch_size = 256 # batch size for mini-batch gradient descent
    weight_decay = 1e-5
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    earlyStop_patience = 40
    learning_rate = 0.01
    D = 2

    model = get_model('Acrobot', use_net, D, device=device)

    if use_net == 'Lagrangian_SinNet':
        earlyStop_patience = 50
        learning_rate = 0.1


    train_loader, valid_loader, input_scaler, output_scaler, = load_train_N_validate_data(join(train_data_path, "data"),
                                                                                          batch_size,
                                                                                          valid_data_path=join(valid_data_path, "data"),
                                                                                          device=device)
    loss_fn = torch.nn.SmoothL1Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    early_stopping = EarlyStopping(patience=earlyStop_patience, verbose=False)

    ### Train model
    model = train(model, train_loader, valid_loader, optimizer, loss_fn, early_stopping, max_training_epoch, goal_loss, is_plot=False)

    ### Get the predict output from test data and save to Matlab file
    train_dataset = load_data_dir(join(train_data_path,'data'), device='cpu', is_scale=False)
    train_input_mat = train_dataset.x_data
    train_output_mat = train_dataset.y_data
    model = model.to('cpu')
    test_dataset = load_data_dir(join(test_data_path,"data"), device='cpu', is_scale=False)
    test_input_mat = test_dataset.x_data
    test_output_mat = predict(model, test_input_mat, input_scaler, output_scaler, 'cpu')
    try:
        mkdir(join(train_data_path,"result"))
    except:
        print('Make directory: ', join(train_data_path,"result") + " already exist")

    # save data as .mat file
    save_result_path = join(train_data_path, "result", use_net+'.mat')
    print('Save result: ', save_result_path)
    sio.savemat(save_result_path, {'test_input_mat': test_input_mat.numpy(),
                                  'test_output_mat': test_output_mat.numpy(),
                                  'train_input_mat': train_input_mat.numpy(),
                                  'train_output_mat': train_output_mat.numpy()})

    test_loss, abs_rms_vec, rel_rms_vec = evaluate_rms(model, loss_fn, test_data_path, input_scaler, output_scaler,
                                                       device, verbose=True)


################################################################################################################


#N_list = [2,3,4,5,6,7,8,9,10,12,15,17,20]
N_list = [20]
std = 1
train_file_list = ['N'+str(i)+'_std'+str(std) for i in N_list]

# use_net_list = ['SinNet', 'ReLuNet', 'SigmoidNet','Lagrangian_SinNet', 'VanillaSinSigmoid_Net','SinLogNet']
# use_net_list = ['VanillaSinSigmoid_Net']
use_net_list = ['SinLogNet']

is_disturb = False
if is_disturb:
    disturb_dir = 'withDisturb'
else:
    disturb_dir = 'withoutDisturb'

for train_file in train_file_list:
    for use_net in use_net_list:
        train_data_path = join("data", "Acrobot","sim", disturb_dir, train_file)
        valid_data_path = join("data", "Acrobot","sim", disturb_dir, "N34_std"+str(std))
        test_data_path = join("data", "Acrobot", "sim",disturb_dir, "N34_std0")
        loop_func(train_data_path, valid_data_path, test_data_path, use_net)

# test
#loop_func('N8_std1', 'Lagrangian_SinNet')