# DAWNBench_Inference
## Resnet101 for DAWNBench inference task on ImageNet

We run Resnet101 on Alibaba Cloud Npu, which consists of 1 NPU.

The following instructions show how to achieve the performance that we submitted to DAWNBench step by step.

0. Get a Npu container from https://promotion.aliyun.com/ntms/act/alinpu.html

1. install python3
```
    wget https://www.python.org/ftp/python/3.5.2/Python-3.5.2.tgz 
    tar -zxf Python-3.5.2.tgz && cd Python-3.5.2 && ./configure && make && sudo make install
```

2. install gcc && pillow && numpy && python-gflags && tensorflow
```shell
    yum -y install gcc+ gcc-c++
    pip3 install pillow numpy python-gflags tensorflow==1.12
    pip3 install hgai (in npu container from https://promotion.aliyun.com/ntms/act/alinpu.html)
```

3. git clone DAWNBench_Inference code
```
   http://gitlab.alipay-inc.com/CognitiveComputing/dawnbench-cognition
```


4. run the following commands to replicate our results submitted to DAWNBench,  
```shell
   cd inference
   tar xzvf model.tar.gz
   ## 1.download tf records of ImageNet to imagenet_val path
   ## 2.run eval 
   sh eval.sh
```

5.the result is as follows:
```shell
top1:0.750200
top5:0.933800
Latency mean:0.387978ms
```
