---
title: 持续学习论文列表（长期更新）
date: 2022-04-10
categories: [科研]
tags: [持续学习, 长期更新]
img_path: /assets/img/
math: true
---


本文


\subsection{History}
\begin{frame}{Background}{History}

The first paper to introduce few-shot learning:

* [Machine Learning, 1997] 
CHILD: A First Step Towards Continual Learning (MARK B. RING, Adaptive Systems Research Group, GMD)

\

% The first combination of deep learning techniques and few-shot learning problems:

% * []


    

\end{frame}




\subsection{Category}
\begin{frame}{Background}{Category}
Replay methods
\begin{itemize}
    \item Rehearsal
    \item Generative (Pseudo Rehearsal)
\end{itemize}




Regularization-based methods


Parameter isolation methods

    

\end{frame}


\section{Replay Methods}
\subsection{Rehearsal}
\begin{frame}{Replay Methods}{Rehearsal}

\textit{Most Notable}

\

* [CVPR 2017] \textbf{iCaRL}: Incremental Classifier and Representation Learning (Sylvestre-Alvise Rebuffi, et al, University of Oxford/IST Austria)

\


Idea: 
\begin{itemize}
    \item Aims to learn a strong data representation
    \item Nearest-Mean-of-Exemplars Classification, examplars take part in prediction
    \item Simply fix memory size, allocate to each task averagely
\end{itemize}

\end{frame}


\begin{frame}{Replay Methods}{Rehearsal}

* [NIPS 2017] (\textbf{GEM}) Gradient Episodic Memory for Continual Learning (David Lopez-Paz, et al, Facebook)

* [ICLR 2018]  Efficient lifelong learning with \textbf{A-GEM} (Arslan Chaudhry, et al, University of Oxford, Facebook)


\


Idea: 
\begin{itemize}
    \item Exploits exemplars to solve a constrained optimization problem
    \item Store previous task gradient
    \item A-GEM propose a small change to the loss function which makes GEM orders of magnitude faster at training time while maintaining similar performance
\end{itemize}

    

\end{frame}


\begin{frame}{Replay Methods}{Rehearsal}
* [NIPS 2019] (\textbf{CLEAR}) Experience replay for continual learning (D. Rolnick, et al, UPenn, DeepMind)

\

Idea:
\begin{itemize}
    \item Actor-critic training on a mixture of new and replayed experiences
    \item Off-policy learning and behavioral cloning from replay to enhance stability
    \item On-policy learning to preserve plasticity
    \item Off-policy: V-Trace learning algorithm
\end{itemize}

    

\end{frame}



\begin{frame}{Replay Methods}{Rehearsal}
* [ICCV 2021] (\textbf{CoPE})  Continual prototype evolution: Learning online from non-stationary data streams (M De Lange, et al, KU Leuven)

    

\end{frame}



\subsection{Generative (Pseudo Rehearsal)}

\begin{frame}{Replay Methods}{Generative}
* [NIPS 2017] (\textbf{DGR}) Continual Learning with Deep Generative Replay (Hanul Shin, et al, MIT, SK T-Brain)

\begin{figure}
    \centering
    \includegraphics[scale=0.3]{img/DGR.png}
\end{figure}

Idea: 
\begin{itemize}
    \item Introduces GAN
    \item Dual model: Generator \& Solver
    \item Data for previous tasks can easily be sampled
\end{itemize}

    

\end{frame}





\section{Regularization Methods}

\begin{frame}{Regularization Methods}
* [ECCV 2016] (\textbf{LwF}) Learning Without Forgetting (Zhizhong Li, et al, University of Illinois Urbana Champaign)



\begin{figure}
    \centering
    \includegraphics[scale=0.4]{img/LwF.png}
\end{figure}


    

\end{frame}




\begin{frame}{Regularization Methods}

\textit{Most Notable}

\

* [PNAS 2017] (\textbf{EWC}) Overcoming catastrophic forgetting in neural networks (James Kirkpatrick, et al, DeepMind, Imperial College London)


\

Idea: 
When training task B after task A, minimize $\mathcal{L}$ instead:

$$
\mathcal{L}(\theta)=\mathcal{L}_{B}(\theta)+\sum_{i} \frac{\lambda}{2} F_{i,i}\left(\theta_{i}-\theta_{A, i}^{*}\right)^{2}
$$

~\\
Constrain important parameters to stay close to their old values

    

\end{frame}


\begin{frame}{Regularization Methods}


* [NIPS 2017] (\textbf{IMM}) Overcoming catastrophic forgetting by incremental moment matching (Sang-Woo Lee, et al, Seoul National University)

\

Idea: incrementally matches the moment of the posterior distribution of the network

 averages the parameters of two networks in each layer, using mixing ratios $\alpha_k$ with
$\sum_k^K \alpha_k = 1$ 




\begin{figure}
    \centering
    \includegraphics[scale=0.6]{img/IMM.png}
\end{figure}



\end{frame}






\begin{frame}{Regularization Methods}
* [IEEE ICPR 2017] (\textbf{R-EWC}) Rotate your networks: Better weight consolidation and less catastrophic forgetting (Xialei Liu, et al, UAB Spain, University of Florence)

\

Idea: A factorized rotation of parameter space in conjunction with EWC

    

\begin{figure}
    \centering
    \includegraphics[scale=0.6]{img/R-EWC.png}
\end{figure}


\end{frame}



\begin{frame}{Regularization Methods}
* [ICML 2017] (\textbf{SI}) Continual learning through synaptic intelligence (Friedemann Zenke, et al, Stanford University)

\

Idea: Bring biological complexity into artificial neural networks

    

\end{frame}




\begin{frame}{Background}{Category}
* [ECCV 2018] (\textbf{MAS}) Memory aware synapses: Learning what (not) to forget (Rahaf Aljundi, et al, KU Leuven, Facebook)

\

Idea: Accumulates an importance measure for each parameter of the net- work, based on how sensitive the predicted output function is to a change in this parameter. When learning a new task, changes to important parameters can then be penalized, effectively preventing important knowledge related to previous tasks from being overwritten. 

    

\end{frame}

\section{Architecture Methods}




\begin{frame}{Architecture Methods}
* [CVPR 2017] \textbf{Expert Gate}: Lifelong Learning with a Network of Experts (Rahaf Aljundi, et al, KU Leuven)



\begin{figure}
    \centering
    \includegraphics[scale=0.3]{img/EG.png}
\end{figure}



\end{frame}


\begin{frame}{Architecture Methods}
* [arxiv 2017] \textbf{PathNet}: Evolution channels gradient descent in super neural networks (Chrisantha Fernando, et al, DeepMind)

\

Idea: 
\begin{itemize}
    \item Agents embedded in network to discover which parameter to re-use for new tasks
    \item Pathways through network are the subset of parameters 
    \item These parameters updated by the forwards and backwards passes of the backpropogation algorithm
\end{itemize}
    

\end{frame}


\begin{frame}{Architecture Methods}
* [CVPR 2018] \textbf{PackNet}: Adding Multiple Tasks to a Single Network by Iterative Pruning (Arun Mallya, et al, University of Illinois at Urbana-Champaign)

\

Idea: 
\begin{itemize}
    \item Network Pruning: free up redundant parameters that can then be employed to learn new tasks
    \item Sequentially “pack” multiple tasks into a single network
\end{itemize}



\begin{figure}
    \centering
    \includegraphics[scale=0.25]{img/PackNet.png}
\end{figure}
    

\end{frame}


\section{Theories of Catastrophic Forgetting}

\begin{frame}{Theories of Catastrophic Forgetting}




\end{frame}
