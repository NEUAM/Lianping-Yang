MED-UN: More Efficient Densely Connected U-Nets for Face Alignment  
======================================================================
Thanks
--------------------------
Thanks to Zhiqiang Dong for the open source code, which has greatly facilitated our work. 
Github:https://github.com/zhiqiangdon/CU-Net  

  
MED-UN structure  
--------------------
For the full flow of data within the network, our model connection mode is dense connection(please refer to the paper for details:
Z. Tang, P. Xi, S. Geng, L. Wu, D. Metaxas, Quantized densely connected u-nets for efficient landmark localization    
       网址：(http://xueshu.baidu.com/usercenter/paper/show?paperid=d0c4220ed2937249a14092a8dc2bb0a3&site=xueshu_se&hitarticle=1)  
Here is a simple illustration:  
    ![](https://github.com/iam-zhanghongliang/MED-UN/blob/master/MED-UN/figures/Dense_connection.jpg)

In order to further improve the performance of the network, we have introduced new ideas: Stacked depthwise separable convolution and fused multi-scale depthwise separable convolution module. Based on the above ideas, we have construct a new block and MED-UN, as shown in the figure below:  
   
!['The_new_block'](https://github.com/iam-zhanghongliang/MED-UN/blob/master/MED-UN/figures/The_new_block.jpg)  
![](https://github.com/iam-zhanghongliang/MED-UN/blob/master/MED-UN/figures/MED-UN(2)_network.jpg)
  
Environment configuration
---------------------
We use python 3.6.2. Pytorch 1.01 during training and testing.


Validate
-------
The first step is to download the 300W test data, URL:https://pan.baidu.com/s/1J0ggq8sRag6KLSIQ3DUOjw  
The second step is to add and the path of the test data MED-UN_model to the validate.py file  
The third step is to run the program: python validate.py --gpu_id 0 --exp_id cu-net-4 --bs 1  
Note: When validating the challenge set, the visualization results are disordered due to the mismatch between the total picture order verified and the test picture order loading, but the visualization results of the challenge subset can be seen on the full set.  



