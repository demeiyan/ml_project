# CNN

* 一个cnn网络结构
![CNN][1]

* All_CNN实现的参数batch_size=256,learning_rate=0.001,weight_decay=0.00015,epoch=300，准确率为82.9%

* 尝试过的参数:

 |learning_rate | weight_decay| | |
 | ------------- | ------------ | --- | --- |
 |0.0001 |   0.0005| | | 
 |0.0003  |  0.00008|
 |0.0004  |  0.00007| 75.5%| 250epoch|
 |0.0005  |  0.001  | 79%|250epoch|
 |0.00005 |  0.00025| 69.450% |	1400 epoch| 
 |0.00008	|0.00015| 72.61%	|300 epoch|
 |0.0005 |	0.00055|||
 |0.0005 |	0.00055 |75.81%	|300 epoch|
 |0.001  |   0.00015 |82.9%  | 300 epoch|
 |0.001  |   0.0001 | 81.8%  | 3000 epoch|
 
* 各类的结果：

 	1. Accuracy of plane : 84 %
	2. Accuracy of   car : 91 %
	3. Accuracy of  bird : 74 %
	4. Accuracy of   cat : 69 %
	5. Accuracy of  deer : 79 %
	6. Accuracy of   dog : 71 %
	7. Accuracy of  frog : 86 %
	8. Accuracy of horse : 86 %
	9. Accuracy of  ship : 89 %
	10. Accuracy of truck : 88 %
	11. Total Accuracy is : 82.160%

[1]: cnn.png
