# ELM-scratch-
ELM(From Scratch)-MNIST dataset

I have attached the file where I've made elm from scratch and used it to predict MNIST digit dataset. 

This was something I was trying to explore on my own to learn about the machine learning model.
Accuracy using 1000(optimum) Hidden neurons was 94% while using 10000 neurons was 97%.
There was obviously Time vs Accuracy trade off. But 1000 neurons was pretty fast with high accuracy.

Some general information on ELM :

=> ELM, belonging to the category of single hidden layer feedforward neural networks, departs from the traditional iterative training methods employed by neural networks, opting for a simpler and more expeditious learning approach.

=> The technique employed by ELM, known as randomized feature mapping, serves to convert input data into a higher-dimensional feature space. By utilizing random hidden neurons with fixed parameters, this mapping process enables the transformation. Additionally, the weights connecting the input layer to the hidden layer are assigned randomly and remain static throughout the training phase, eliminating the need for adjustments.

=> An area of particular focus for ELM lies in the efficient training of output weights, achieved through the solution of a linear system of equations. This approach facilitates swift learning and mitigates the computational complexity commonly associated with iterative methods utilized in traditional neural networks. Notably, ELM exhibits commendable generalization capabilities, enabling its efficacy in diverse machine learning tasks such as classification, regression, and more.

=> To summarize, ELM stands as a rapid and efficient machine learning algorithm, characterized by its employment of a single hidden layer feedforward network. Leveraging the randomized feature mapping technique and the solution of linear equations to train output weights, ELM emerges as a solution suitable for a wide array of applications, boasting both simplicity and computational efficiency.



General Information On MNIST dataset:

=> MNIST Dataset: The MNIST dataset is a widely-used benchmark dataset in the field of machine learning. It consists of a collection of handwritten digits, ranging from 0 to 9, represented as 28x28 pixel grayscale images.

=> Image Classification: The main purpose of the MNIST dataset is to facilitate image classification tasks. Machine learning models are trained on this dataset to learn how to recognize and classify handwritten digits accurately. It serves as a foundation for developing and evaluating various image classification algorithms and techniques.

=> Dataset Composition and Size: The MNIST dataset is composed of 60,000 training images and 10,000 test images. These images are divided into ten classes, each representing one of the ten possible digits. The dataset's balanced nature ensures an equal number of samples for each class, promoting fair evaluation and comparison of classification models.
