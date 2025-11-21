# An-improve-on-ResNet-Introduces-dynamic-connections
When I was learning about ResNet, I realized that the shortcut connections were manually designed. This gave me an idea: why not make the connections dynamic and allow them to "grow" on their own? I call it **DResNet**. *(It's unfortunate that I didn't know the DenseNet at that time)*  

So, I introduced a weight array for each layer and designed the connections based on this weight array, allowing the weights to be learned in a way that enable the connections to "grow" autonomously. Although my teacher said it was useless, it was my first original idea during my student life, so I uploaded it to GitHub.

I'll introduce my idea by this directory structure：
- Basic Frame
  - The Base Frame
  - The Change
- Dynamic Connection
  - Weight Array
  - Activativation Function for Weight Array
    - Sigmoid Function
    - Gauss Function
- Comparason
  - Loss and Accuracy
    - All models' comparason
      - Loss and Acc
    - Each model's training result
      - Train and test, val
  - Parameters
  - Activation Function
- Code
  - The Structure of Project
  - Run Code


## Basic Frame
My changes are based on the ResNet framework, so you can see that my network is almost like ResNet.

### The Base Frame
The basic framework of my net is ResNet and I didn't change much of it. Based on ResNet, The layers are connected by shortcuts, and beyond that, layers anywhere in the network can freely connect to other layers. Suppose the input port is at the rear and the output port is at the front. All layers except the last two layers (because there no more layers before them) can build connection to other layers before them.

### The Change
For each layer, I assign a **Weight Array**.   

Consider a network with n+2 layers. According the ResNet, it has one conv-layer and one fc-layer, so the body has n layers.   

In the n layers, we consider the i-th layer, It has a **Weight Array** of lenth **i-2**, meaning it can freely connect the rear **i-2** layers. *(For example, the 3rd layer: the layers behind it are the 1st layer and the 2nd layer. If it wants to connect to the rear layers, it can only connect the 1st layer.For 2nd layer, it must connect to the 3rd layer. So the 3rd layer can only freely choose to connect one layer, according to the format: **i-2**, which is 3-2 = 1)*  

*(Anyway, About the connection method: Of cause, you can naturally think of forward connections, where a layer's output gose to subsequent layers, that's natural, **Right?** But, when implementing the code, I found that having the i-th layer choose which rear layers to connect to is more earier to implement and is equivalent to the more natural idea.)*  

The i-th layer can choose the output (x + F(x)) from the rear **i-2** layers to add into i-th layer's input. Considering potential shape differences, I designed it so that the rear output must go through a convolutional layer if there's a difference between it's output and the input shape, and even when the shapes are the same, the convolution still exists.  

## Dynamic Connection  

### Weight Array  

After the convoluton, The **Weight Array** can be used. The output which has the same shape as the i-th layer's input, must be multiplied by the corresponding **F(Weight)** which from the i-th layer's **Weight Array**. *(F is a function to realize the method of choose. If F is f(x)=x, the F(Weight)=Weight. The choice of the function will be discussed in **Dynamic connection -- Activation function of Weight Array**)*   
  
So, The i-th input is:  

$${\color{red} x_i = x_{i-1}+ \sum_{k=1}^{i-2}F(w_k)\times y_k}$$  
$${\color{red} y_k = x + f(x)}$$  

*(y_k is the output of a Residual layer)*  

Based on the value of the **Weight**, The selection machanism is obvious. If the i-th layer don't need the j-th layer's output ($j \in {1, 2, 3 ... i-2}$), the value of the corresponding **F(Weight)** will be zero or near zero, which means the i-th layer don't need this layer's output. The F(Weight) changes according to the F function.  

### Activation Function for Weight Array  

An appropriate **Activation Function** is important.  
1. Fisrt, The range of the Function shouldn't exceed 1 or go below 0 ($range \in [0,1]$). *(Well, Maybe you can use the f(x)=x, but I don't recommand)*
2. Second, The nature of the Function will influence the network's convergence rate and the accuracy. *(The differences will be discussed in the next part)*  

Based on the above two relus, I finally chose two functions.  

They're the **Sigmoid Function** and **Gauss Function**.  

*(I want to show the images of the Sigmoid and Gauss functions, But I don't know how to display them. My apologies)*  

#### Sigmoid Function  

Anyway, Though I can't show the images, I think everyone knows them *Right?*  (>_<|||). Here's code to generate the images of the Sigmoid and the Gauss functions.    


    import numpy as np
    from matplotlib import pyplot as plt
    
    Sigmoid_Function = lambda x: 1 / (1 + np.exp(-x))  # Sigmoid Function
    Gauss_Function = lambda x, k: np.exp(-k * (x ** 2))  # Gauss Function
    
    x = np.arange(-100,100,0.1)
    sigmoid_y = [Sigmoid_Function(_) for _ in x]
    gauss_y = [Gauss_Function(_, 10) for _ in x]
    
    ax1 = plt.subplot(2,1,1)
    plt.plot(x,sigmoid_y)
    ax1.set_title('Sigmoid')
    
    ax2 = plt.subplot(2,1,2)
    plt.plot(x,gauss_y)
    ax2.set_title('Gauss')
    
    plt.tight_layout()
    plt.show()
  

  
When using the **Sigmoid Function** as the activation function, during training, the **Weight**s in the **Weight Array** will become more and more convergent. Eventually, even if the **Weight** change somewhat, the **F(Weight)** won't change much, leading to the connections become 'solid'.   

Though the fast convergence make the net display the better convergence during training, and make the accuracy higher. But I think the rapid convergence may cause the model to get stuck in a **Local Optimal Solution**. Anyway, using **Sigmoid** as the activation function(*which can also be called the 'Choose Function'*) is useful accodring to my experiments on CIFAR-10, CIFAR-100, Tiny-ImageNet *(also make me poor and tired! (´Д｀))*  

#### Gauss Function  

Considering that the **Sigmoid Function** can make convergence faster but has the possibility of getting stuck in local optimal soulution. So, I conceived a new function, **Gauss Function**. The formula for the **Gauss Function** is: **$y=e^{-kx^2}$**.  

If you have seen the image of the **Gauss Function**, you will find that the low probability of high-valued dependent variables perfectly accords with the dynamic connection concept *(Eh, it's just my opinion.)*. The function's sharp profile gives the networks more opportunities to jump out of the local optimal solutions.  

And, The Gauss function's parameter of '*k*' is important, because it is trainable in my idea. The '*k*' can make the nwtwork to disjust the ability of the '*sharp*'.  

But, the disadvantages are obvious, The network's convergence procedure may be **highly fluctuating**. *(Eh, When I write here, I haven't done the Gauss function experiment. It's expect to be finished when I have done the **Comparason-Parameters-Gauss Function or -Loss and Accuracy**)*.  

## Comparason  

### Loss and Accuracy  

Each model's independent training result images are stored in **Result/Tiny ImageNet/{model}**. Each file has two images: **{model}_acc** and **{model}_loss**. The model below is an 18-layer model, which has one conv-layer, an 18-layer body and one fc-layer.  

#### All models' comparason  

In **Result/Tiny ImageNet**, there're two pictures: **acc.png** and **loss.png**.  

##### Loss and Acc  

In **loss.png**, the loss of ResNet and my network called **DResNet** goes down quickly, and then increase around epoch 25. When the loss converges, the ResNet's loss is higher than CNN's, but DResNet's is lower than ResNet's, which means DResNet can achieve better convergence after training and ease the problem of high post-convergence loss.  

<img width="1920" height="1440" alt="loss" src="https://github.com/user-attachments/assets/488fb8b8-ab64-4605-82ec-26818217aab3" />  

In **acc.png**, we can see that the accuracy of ResNet and DResNet both is higher than CNN's. At 25 epochs, DResNet shows better convergence performance and higher accuracy.  

<img width="1920" height="1440" alt="acc" src="https://github.com/user-attachments/assets/6673dfc3-36f1-4c8c-ae0c-16bfa0915882" />  

#### Each model's training result  

##### Train and test, val  

During training, All three models show increasing accuracy until the test accuracy stabilizes. But CNN's train acc curve grows slowly and stops at around **70%**. In contarst, ResNet and DResNet stop at 85% and 100% respectively. Well, the overfitting capability of these two models might be *Good?*  

### Parameters  

The parameters of CNN is 11,101,512, and the parameters of ResNet is 11,627,272, and the parameters of DResNet is 12,320,932. Although the dynamic connections have $O(n^2)$ complexity, the parameter increase is not substantial.  

### Activation Function  

The choice of activation function can cause a big difference, *I guess*. If I have time, I'll test the **Gauss Function**.  

## Code  

### The Structure of Project  

The Project is structured as three components: **Run.py**, **Model (folder)**, **Result (folder)**.  

The models' code is placed in **Model** folder. And the **Result** contains all the images for the three models, and the overall comparison of **Accuracy** and **Loss**. You can run the code using **Run.py**.  

### Run Code  

The comparison interface is in **Run.py**, you can run the code in this file. *(I don't know how to use the Commmand Line to provide more options for users, So I give a main interface directly)*
