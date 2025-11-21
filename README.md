# An-improve-on-ResNet-Introduces-dynamic-connections
When I was learning about ResNet, I realized that the shortcut connections were manually designed. This gave me an idea: why not make the connections dynamic and allow them to "grow" on their own? I call it **DResNet**. *(It's bad that I don't know the DenseNet at that time)*  

So, I introduced a weight array for each layer and designed the connections based on this weight array, allowing the weights to be learned in a way that would enable the connections to "grow" autonomously. Although my teacher said it was useless, it was my first original idea during my student life, so I uploaded it to GitHub.

I'll introduce my idea by this directory：
- Basical Frame
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


## Basical Frame
My change is base on the ResNet frame, so you can find my net is almost like the ResNet.

### The Base Frame
The base frame of my net is ResNet and I don't change much of the ResNet. Base on ResNet, The layers are connected by shortcut, and beyond it, The layers nowhere in the net can freely connect other layers.Suppose the input port is the rear and the output port is front. All layers except the last two layers (because there no more layers before them) can build the connection to other layers before it.

### The Change
For each layers, I give it a **Weight Array**.   

Consider a n+2 layers' net, according the ResNet, it has one conv and one fc layer, so the body of it is n layers.   

In the n layers, we consider the i-th layer, It have a **Weight Array** which lenth is **i-2**, meaning it can freely connect the rear **i-2** layers. *(such as the 3-th layer, the layers rear it are the 1-th layer and the 2-th layer, if it want to connect the rear, it can only connect the 1-th layer.For 2-th layer, it must connect the 3-th layer. So the 3-th layer can only freely choose to connect one layer, according the format: **i-2**, it's 3-2 = 1)*  

*(Anyway, There's something about the connection method: Of cause you can naturally think out the front connection, for a layer, its output can give the front layer, it's nature **Right?** But, when realizing the code, I find that we should make the i-th layer to choose the rear layer to connect, It's more earier to realize and it's equivalent to the more nature idea.)*  

The i-th layer can choose the output(x + F(x)) from the rear **i-2** layers to add into i-th layer's input. Consider the difference of shapes, I design that the rear output must through a convolutional layer if there's a difference between the output and the shape of the input, and even the both shapes are same the convolution still exist.  

## Dynamic Connection  

### Weight Array  

After the convoluton, The **Weight Array** can be used. The output which is the same shape as the i-th layer's input must multiplied by the corresponding **F(Weight)** which from the i-th layer's **Weight Array**. *(the F is a function to realize the method of choose. If the F is f(x)=x, the F(Weight)=Weight. The choose of the function will be said in **Dynamic connection -- Activation function of Weight Array**)*   
  
So, The i-th input is:  

$${\color{red} x_i = x_{i-1}+ \sum_{k=1}^{i-2}F(w_k)\times y_k}$$  
$${\color{red} y_k = x + f(x)}$$  

*(y_k is the output of a Residual layer's)*  

According the value of the **Weight**, The use of choice is obvious. If the i-th layer don't need the j-th layer's output ($j \in {1, 2, 3 ... i-2}$), the value of the corresponding **F(Weight)** will be the zero or near zero, which means the i-th layer don't need this layer's output. The F(Weight) is change with the change of the F function.  

### Activation Function for Weight Array  

An appropriate Activation Function is important.  
1. Fisrt, The range of the Function shouldn't beyond 1 or below 0 ($range \in [0,1]$). *(Eh, Maybe you can use the f(x)=x, but I don't recommand)*
2. Second, The nature of the Function will influence the net's convergence rate and the accuracy. *(The difference will be said in next part)*  

Through the above two relus, I choose two functions finally.  

It's **Sigmoid Function** and **Gauss Function**.  

*(I want to show the images of the Sigmoid and the Gauss, But I don't know how to display. My apology)*  

#### Sigmoid Function  

Anyway, Though I can't show the images, but I think everyone knows it *Right?*  (>_<|||). There're code to generate the images of the Sigmoid and the Gauss.    


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
  

  
When use **Sigmoid Function** as the activation function, during the training, the **Weight** which in the **Weight Array** will become more and more convergent. At last, even the **Weight** has some changes, the **F(Weight)** won't change much leading to the connection become 'solid'.   

Though the fast convergence make the net display the better convergence during training, and make the accuracy higher. But I think the rapid convergence will caused the model get into a **Local Optimal Solution**. Anyway, using **Sigmoid** as the activation function(*also can call it 'Choose Function'*) is useful accodring my experiment on CIFAR-10, CIFAR-100, Tiny-ImageNet *(also make me poor and tired! (´Д｀))*  

#### Gauss Function  

Consider the **Sigmoid Function** can make the convergence reach faster but have possibility of getting into local optimal soulution. So, I conceive a new function, **Gauss Function**. The formation of **Gauss Function** is: **$y=e^{-kx^2}$**.  

If you have watched the image of the **Gauss Function**, you will find that the low proporation of the high value dependent variables is perfectly accord with the dynamic connection *(Of cause, it's my opinion.)*. The function's image is sharp making it can give the net more chances to jump out of the local optimal solution.  

But, the disadvantages are obvious, The net's convergence procedure may be **fluctuation-intense**. *(Eh, When I write here, I haven't do the Gauss function experiment. It's expect to be finished when I writing the **Comparason-Parameters-Gauss Function or -Loss and Accuracy**)*.  

## Comparason  

### Loss and Accuracy  

The each models' independent training result images are stored in **Result/Tiny ImageNet/{model}**. Each file has two images of **{model}_acc** and **{model}_loss**. The model below is 18-layers model, which has one conv-layer,18-layer body and one fc-layer.  

#### All models' comparason  

In **Result/Tiny ImageNet**, there're two pictures of **acc.png** and **loss.png**.  

##### Loss and Acc  

In **loss.png**, the loss of ResNet and my net called **DResNet** goes down fastly, and then goes up near the 25-epoch. And, when the loss is convergent, the ResNet's loss is higher than the CNN, but the DResNet is lower than the ResNet, which means the DResNet can be more convergence after training and ease the problem of the high after-convergent loss.  

<img width="1920" height="1440" alt="loss" src="https://github.com/user-attachments/assets/488fb8b8-ab64-4605-82ec-26818217aab3" />  

In **acc.png**, we can find that the accuracy of the ResNet and the DResNet both are higher than the CNN. And, at 25 epochs, the DResNet has a better convergence performance and the higher accuracy.  

<img width="1920" height="1440" alt="acc" src="https://github.com/user-attachments/assets/6673dfc3-36f1-4c8c-ae0c-16bfa0915882" />  

#### Each model's training result  

##### Train and test, val  

During the training, Three models are getting higher after the test accuracy become stable. But, the CNN' train acc curve is grow slowly, and the value stops at near **70%**. However, the ResNet and the DResNet stop at 85% and 100%. Eh, the overfitting ability of two models may be *Good?*  

### Parameters  

The parameters of CNN is 11101512, and the parameters of ResNet is 11627272, and the parameters of DResNet is 12320932. Though the dynamic connection is $O(n^2)$, the increment is not much.  

### Activation Function  

The choice of activation function can cause a big difference, *I guess*. If I have time, I'll test the **Gauss Function**.  

## Code  

### The Structure of Project  

The Project will design as three files: **Run.py**, **Model (folder)**, **Result (folder)**.  

The models' code will put in **Model** folder. And the **Result** wil put entire pictures of three models, and the total comparason of **Accuracy** and **Loss**. You can run the code with **Run**.  

### Run Code  

The interface of the comparason is **Run.py**, you can run the code in this file. *(I don't know how to use the Commmand Line to give more choice for users, So I give a main interface directly)*
