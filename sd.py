from model import DifferNet
from utils import load_datasets, make_dataloaders, preprocess_batch
import cv2
import os
import random
import config as c
import scipy
import numpy

#print('train_loader length:', len(train_loader))
#dataiter = iter(train_loader)
#images, labels = dataiter.next()
#print('images.shape:', images.shape, ',labels.shape:', labels.shape)


def get_sd(dataset_path, class_name, defectGen=False):
	if(defectGen):
		#1. simple defect generation
		data_dir_train_good = os.path.join(dataset_path, class_name, 'train/good')
		data_dir_train_fake_ng = os.path.join(dataset_path, class_name, 'train_fake_ng/fake_ng')
		if(os.path.isdir(data_dir_train_fake_ng) is False):
			os.makedirs(data_dir_train_fake_ng)

		onlyfiles = [f for f in os.listdir(data_dir_train_good) if os.path.isfile( os.path.join(data_dir_train_good, f))] 
		for file in onlyfiles:
			source_file_dir = os.path.join(data_dir_train_good,file)
			dest_file_dir   = os.path.join(data_dir_train_fake_ng,file)
			img1 = cv2.imread(source_file_dir)
			
			#get defect size
			defect_size_max = round(img1.shape[0] * 0.3) #30%
			defect_size_min = round(img1.shape[0] * 0.05) #5%
			h,w = random.sample(range(defect_size_min,defect_size_max),2)
			#get defect position
			x,y,x2,y2 = random.sample(range(0, (img1.shape[0]-max(h,w))),4)
			tmp = img1[x:(x+w),y:(y+h),:]
			img1[x2:(x2+w),y2:(y2+h),:] = tmp
			cv2.imwrite(dest_file_dir, img1)
			print('src:', source_file_dir, 'dest:', dest_file_dir ,'shape', img1.shape)

	#2. make dataloader
	train_set,test_set,fake_ng_trainset = load_datasets(dataset_path, class_name, False)
	train_loader,_ = make_dataloaders(train_set, test_set, False)
	fake_ng_train_loader,_ = make_dataloaders(fake_ng_trainset, test_set, False)

	#file order check
	for i in range(len(train_set)):
		tr_ok, _ = train_loader.dataset.samples[i]
		tr_ng, _ = fake_ng_train_loader.dataset.samples[i]
		print('tr_ok:',tr_ok,'tr_ng:',tr_ng)
	
	#3. get features
	model = DifferNet([0])
	model.to(c.device)
	model.eval()
	train_dataiter = iter(train_loader)
	fake_ng_train_dataiter = iter(fake_ng_train_loader)

	dims_dic = {}
	dims_selected_dic = {}
	for x in range(len(train_loader)):
		# inputs1, _ = preprocess_batch(train_dataiter.next())
		# inputs2, _ = preprocess_batch(fake_ng_train_dataiter.next())
		inputs1, _ = preprocess_batch(next(train_dataiter))
		inputs2, _ = preprocess_batch(next(fake_ng_train_dataiter))
		features1 = model.get_features(inputs1)
		features2 = model.get_features(inputs2)
		print('c.feature_sd:', c.n_feat_sd, 'inputs1.shape:', inputs1.shape, 'features1.shape:',features1.shape)
		print('c.feature_sd:', c.n_feat_sd, 'inputs2.shape:', inputs2.shape, 'features2.shape:',features2.shape)

		batch_size = features1.shape[0]
		full_dim_size = features1.shape[1]
		for i in range(batch_size):
			for j in range(full_dim_size):
				f1 = features1[i,j,:,:].detach().numpy().reshape(-1)
				f2 = features2[i,j,:,:].detach().numpy().reshape(-1)
				#print('f1==f2:',sum(f1==f2),'f1.shape:',f1.shape)
				pvalue = scipy.stats.ttest_ind(f1, f2, equal_var=False)
				#print('data index:',i,'dim no:',j,'pvalue:',pvalue.pvalue,'feature1.shape:',features1[i,j,:,:].shape,'f1.shape:',f1.shape)
				
				if j in dims_dic:
					dims_dic[j] = dims_dic[j] + [pvalue.pvalue]
				else:
					dims_dic[j] = [pvalue.pvalue]
				
				if pvalue.pvalue <= 0.05:
					if j in dims_selected_dic:
						dims_selected_dic[j] += 1
					else:
						dims_selected_dic[j] = 1


	# for x in range(len(dims_dic)):
	# 	print('dim idx:',x,'p-value mean:',numpy.mean(dims_dic[x]))
	# for x in range(full_dim_size):
	# 	if x in dims_selected_dic:
	# 		print('dim:',x,'selected cnt:',dims_selected_dic[x])
	# 	else:
	# 		print('dim:',x,'selected cnt:',0)

	sd_dic = dict(sorted(dims_selected_dic.items(), reverse=True, key=lambda item: item[1]))
	print(sd_dic)

	sd_list = []
	for x in sd_dic:
		sd_list = sd_list + [x]
	
	print('unsorted dim:',sd_list)
	sd2 = sd_list[:c.n_feat_sd]
	print('unsorted sd:',sd2)
	sd2.sort()
	print('sorted sd:',sd2)
	return sd2

	#4. get dims

'''
def getDims1(self, dl, p_value_th, td_size, rd_size):
    full_dims = list(range(0,td_size))    
    diff_ok = self.getEmbedding(dl,full_dims,0)
    diff_ng = self.getEmbedding(dl,full_dims,0,defectGeneration=True)
    selected_dims = []        
    for j in range(0,diff_ok.shape[0]):
        selected_dim = {}
        for i in range(0,td_size):
            data_ok = diff_ok[j,i,:].numpy().reshape(-1)
            data_ng = diff_ng[j,i,:].numpy().reshape(-1)
            pvalue = scipy.stats.ttest_ind(data_ok, data_ng, equal_var=False)
            if( pvalue.pvalue < p_value_th):
                selected_dim[i] = pvalue.pvalue
        sorted_dims = sorted(selected_dim.items(),reverse=False,key=lambda x:x[1])
        selected_dims += [x[0] for x in sorted_dims[:rd_size]]
    
    dims_dic = {}
    for i in range(0,td_size):
        dims_dic[i] = selected_dims.count(i)
    sorted_dims = sorted(dims_dic.items(),reverse=True,key=lambda x:x[1])
    sd = [ x[0] for x in sorted_dims[:rd_size] ]
    return sd	
'''    
