---
title: "OpenCoach: Genetic Algorithms for Martial Arts & Physical Training"
date: 2022-09-11T14:28:15+01:00
draft: false
---

## Abstract

![Abstract Cover Image](https://images.unsplash.com/photo-1655720406770-12ea329b5b61?ixlib=rb-1.2.1&ixid=MnwxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8&auto=format&fit=crop&w=3132&q=80)

This white paper looks to investigate the possibilities current AI/ML technologies can offer us regarding physical health and training, with a focus towards various martial arts. In the first part of this paper, we will look at the current state of the art. We will see some current ML/AI models and their learning environments and look at how these can be manipulated to work in the context of physical training. 

We then will describe the technical approach, how these systems/technologies can be connected to train our system to perform various martial arts. We will discuss matters such as how we reward a model for physical tasks, how genetic algorithms can be used to improve upon these techniques and what methods could improve the ML (Machine Learning) training for our specific tasks.  

To conclude, we will discuss how participants can use this software. This includes how to develop a Computer Vision system to analyze our posture, how we measure the accuracy of a user's pose against the computer models and what the user gains from this technology over traditional training. 

## Part One: The State of the Art

![State of the Art Cover Image](https://images.unsplash.com/photo-1655720407000-a3e5067cdfb2?ixlib=rb-1.2.1&ixid=MnwxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8&auto=format&fit=crop&w=3132&q=80)

> "Where there is preparation, there is no fear." - Hwang Kee

Given the words of Hwang Kee above, for our studies it is important to identify the best 'teacher' for us to fully develop our study. In our case, this means finding the right models and papers in a world of ever-growing ML/AI applications. 

Meta has recently investigated the idea of simulating sports in a recent paper [¹](#references). Their focus is towards two-player sports that are competitive, where the boxing models were rewarded primarily for closeness of a strike and less so for energy and if the model was facing the opponent. These parameters proved to work well, but perhaps there could be a more general set of parameters that work for multiple sports. This paper also shows the possibility to train a rigged skeleton on spatial data to accomplish a physical task, suggesting this could be transferred for use with our exercise environment. 

Furthermore, Nvidia [²](#references) has recently studied the use AI (Artificial Intelligence) can have in animation synthesis. This has led them to create programs to simulate virtual worlds with years of training in a fraction of that time. In this paper, they simulated 10 years' worth of learning in approximately ten days. This could be used in our case alongside Meta’s research to speed up the time it takes for a trainer to teach a model of a recent activity, allowing for new regiments to be added on a near-weekly basis when given the correct heuristic. 

This heuristic can be obtained from research done by both Google [³](#references) and Gines Hildago [⁴](#references). Both efforts look to track a 3D skeletal posture with Computer Vision. Tracking these points in 2D is trivial, while 3D tracking is more complex. This feature would allow for both the model to watch a trainer perform an activity to learn itself, and then watch a user attempt the same to compare how accurately they are performing it. 

Unity has also created resources to facilitate this type of 3D learning in their 'ML-Agents' toolkit [⁵](#references), which uses the PyTorch framework as a backbone. This would allow for the simple creation of different learning environments for several types of sports, and even in future studies provide trainers with an interface to create their own contexts to train for unforeseen activities for this. 

## Part Two: The Technical Approach

![Technical Approach Cover Image](https://images.unsplash.com/photo-1655720408861-8b04c0724fd9?ixlib=rb-1.2.1&ixid=MnwxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8&auto=format&fit=crop&w=3132&q=80)

> "Train tirelessly to defeat the greatest enemy: yourself, and to discover the greatest master: yourself." - Shi Su Yan

As the quote above hints, much of the technical explanation will surround how the AI will learn to become better than its past iterations. Foremost, we need a dataset to train the model on, as at this point we do not have the software developed to intake a live feed of motions. Therefore, the TUHAD dataset will serve as our base test for the A.I (Artificial Intelligence). We will take the motion data from this collection and use it as our training and testing data for the model. Given that there is only one way to perform these activities in the dataset, we will vary the input by mutating each key point by a given amount, which will correlate to a similar but not identical move. 

To train the A.I, we must build its environment first in Unity. Now given the context is Taekwondo, a striking martial art, this environment will contain sensors on virtual striking pads and dummies. These measurements will act as one of many parameters by which the model needs to optimize. Others may include balance, speed, and safety of position. The latter of these can only be measured in a competitive context, whereby when one virtual fighter uses a move on an opponent it is measured how often this will result in the model taking some form of damage. 

With this setup, we implement the time-bending technology that Nvidia provides, and we let our model train over many generations. This learning is done by use of a genetic algorithm where the models that are most successful at the task continue to the next generation. Mutations as mentioned before arise from offsetting some of the 3D coordinates on the skeleton to generate a new type of move. Telemetry can be used to monitor this progress live by the trainer and allow for adjustments mid-training, but this is for future research. 

Once this technology has been proven to show satisfactory results, the next step is to make this technology accessible to traditional trainers. This would be done by creating a dashboard that lets the trainer enter information on their new task to a group and will record them performing it both as a video and the posture tracked by the MML (Machine Learning) models seen before. Using the software provided by Google, these 3D coordinates will become the new input to our ML (Machine Learning) model, allowing for it to be trained on any martial art we desire. 

## Part Three: The User Experience

![User Experience Cover Image](https://images.unsplash.com/photo-1655720410101-c5cc15b1faf0?ixlib=rb-1.2.1&ixid=MnwxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8&auto=format&fit=crop&w=3132&q=80)

> "If you want to be a champion, you need to train like one." - George St. Pierre

Now that we have our "champion" model, which has practiced a technique thousands of times, it is time we learn from this model. Most critically, we need to develop some form of measurement to see how accurately a user imitates the model in each task. For this white paper, we will stick with a simple approach, tracking both the models and users' joint positions and getting the absolute difference between the two. If the average of these points falls below a given threshold, we will consider the technique as incorrect, and given the points with the largest difference can make suggestions as to how to improve. Given the model's points can be tracked easily, we now need to figure out how to track the users' correlated points. 

Google's "MediaPipe" project is the perfect framework for this. This model is trained to take a video feed of a human performance and track predefined points on the skeleton in a 3D context for each frame. This model can then return the data for use in other applications, in our case it will act as the input to the ML (Machine Learning) model we defined above, where we feed the co-ordinates to a skeleton rigged in Unity and use this as the heuristic for our genetic algorithm. This way, we can accelerate the improvement of this task as we do not need the model to learn the basics of the technique from scratch. 

From here, we need to connect both the Unity Learning Environments with this front-end wrapper of the MediaPipe library. This can be done by creating a hybrid application, in which we have the MediaPipe application and a WebGL compiled version of the Unity Learning Environments. This allows us to feed the points tracked in our JavaScript MediaPipe and feed them into the Unity software, all with a friendly UI/UX. This allows any trainer to easily make use of the software without needing technical knowledge. 

## Conclusion

![Conclusion Cover Image](https://images.unsplash.com/photo-1655720406100-3f1eda0a4519?ixlib=rb-1.2.1&ixid=MnwxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8&auto=format&fit=crop&w=3732&q=80)

As we have seen, the technology currently exists to facilitate the creation of Machine Learning models to learn physical training tasks and compare these models against humans during practice to provide them live feedback on their posture and how they can improve their technique. 

I hope to investigate these possibilities further through a Research Masters in the near future. This would allow for the time to improve upon the software being actively worked on to facilitate improved learning, environments, and rewards metrics to improve the model's effectiveness. Further, this could be expanded to other physical activities not mentioned in this white paper, such as soccer and tennis.  

In the future, I hope this technology can be used to allow anyone to learn whatever physical activities they wish from whatever place they may be. I also hope this will act as a supplement for trainers to offer their teams so that they can work in a personalized training course, rather than more general ones found currently online. Hopefully, given the technology, state of the art and our discussion above, this type of technology will become commonplace in the next couple of years. 

## References

[1] https://research.facebook.com/publications/control-strategies-for-physically-simulated-characters-performing-two-player-competitive-sports/ \
[2] https://nv-tlabs.github.io/ASE/ \
[3] https://research.google/pubs/pub48292/ \
[4] https://arxiv.org/abs/1812.08008 \
[5] https://github.com/Unity-Technologies/ml-agents 