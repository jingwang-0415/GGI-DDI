import  torch
import time
import os

class log_util():
    def __init__(self,flag,radio):

        timestamp = time.strftime("%Y%m%d_%H%M%S")
        self.flg = flag
        model_save_dir = os.path.join('save', self.flg)
        name1= timestamp+'radio_{}'.format(radio)
        self.name2 = self.flg+name1+'SSA_cold_start'
        self.model_save_dir = os.path.join(model_save_dir,name1)
        self.log_dir = os.path.join(self.model_save_dir,'log')
        self.para_dir = os.path.join(self.model_save_dir,'model')
        self.glob_metric = {
            'acc':0,
            'auc':0,
            'f1':0,
            'pre':0,
            'recall':0,
            'ap':0,
            'auroc':0
}
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)
            os.mkdir(self.para_dir)
    def compare(self,local_metric,flag,epoch,model):
        before_best = self.glob_metric[flag]
        if local_metric>before_best:
            # self.save_model(model,epoch)
            self.glob_metric[flag] = local_metric
    def metirc_compare(self,acc,auc,f_1,pre,re,ap,auroc,epoch,model):
        self.compare(acc,'acc',epoch,model)
        self.compare(auc,'auc',epoch,model)
        self.compare(f_1,'f1',epoch,model)
        self.compare(pre,'pre',epoch,model)
        self.compare(re,'recall',epoch,model)
        self.compare(ap,'ap',epoch,model)
        self.compare(auroc,'auroc',epoch,model)

    def save_log(self,epoch,acc,auc,f_1,pre,re,ap,auroc,type,model,opt):
        file_handle1 = open(self.log_dir+'/{}.txt'.format(self.name2), 'a')
        file_handle1.write('type:%s ' % type)
        file_handle1.write('\n')
        file_handle1.write('epoch:%d ' % epoch)
        file_handle1.write('AUC:%f ' % auc)
        file_handle1.write('Pre:%f ' % pre)
        file_handle1.write('AUPR:%f ' % ap)
        file_handle1.write('ACC:%f ' % acc)
        file_handle1.write('f1:%f ' % f_1)
        file_handle1.write('recall:%f ' % re)
        file_handle1.write('auroc:%f' % auroc)
        file_handle1.write('\n')
        file_handle1.close()
        self.save_model(model,opt, epoch)

        if type != 'train':
            # self.save_model(model,opt,epoch)
            self.metirc_compare(acc,auc,f_1,pre,re,ap,auroc,epoch,model)
            if epoch % 10 == 0:
                file_handle1 = open(self.log_dir + '/{}.txt'.format(self.name2), 'a')
                file_handle1.write('type:%s ' % 'glob_metic')
                file_handle1.write('\n')
                file_handle1.write('epoch:%d ' % epoch)
                file_handle1.write('AUC:%f ' % self.glob_metric['auc'])
                file_handle1.write('Pre:%f ' % self.glob_metric['pre'])
                file_handle1.write('AUPR:%f ' % self.glob_metric['ap'])
                file_handle1.write('ACC:%f ' % self.glob_metric['acc'])
                file_handle1.write('f1:%f ' % self.glob_metric['f1'])
                file_handle1.write('recall:%f ' % self.glob_metric['recall'])
                file_handle1.write('auroc:%f ' % self.glob_metric['auroc'])

                file_handle1.write('\n')
                file_handle1.close()

        
    def save_model(self,model,opt,epoch):
        checkpoint = {
            'model':model.state_dict(),
            'opt':opt.state_dict()
        }
        torch.save(checkpoint,
               self.para_dir+'/{}_checkpoint.pt'.format(epoch))
if __name__ == '__main__':
    pass
    # save_log(0,0,0,0,0,0,0,0,0)
    # print(para_dir+'/{}_{}_checkpoint.pt'.format(0,0))