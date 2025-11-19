# An-improve-on-ResNet-Introduces-dynamic-connections
When I was learning about ResNet, I realized that the shortcut connections were manually designed. This gave me an idea: why not make the connections dynamic and allow them to "grow" on their own? *(Baddly, I don't know the DenseNet at that time)*  

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
  - Parameters
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

Though the fast convergence make the net display the better convergence during training, and make the accuracy higher. But I think the rapid convergence will caused the model get into a **Local Optimal Solution**. Anyway, using **Sigmoid** as the activation function(*also can call it 'Choose Function'*) is useful accodring my experiment on CIFAR-10, CIFAR-100, Tiny-ImageNet *(also make me poor! (´Д｀))*  

#### Gauss Function  
