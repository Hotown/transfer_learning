# Transfer Learning 论文总结

## 阅读论文时的问题设置

1. **What is the problem addressed in the paper?**

   > The answer shall address: what are the input $X$ (e.g. a single RGB image, an image sequence, or an RGBD imagel, what are the Y (e.g., pose of the human in the image) what are the constraints on $\mathrm{X}$ and/or $\mathrm{Y}$, if any.

2. **Is this a new problem?**

  + **If it is a new problem, why does it matter?**
    
    > A new, meaningful, and yet challenging research problem needs a keen eye to spot, and if it is big/important enough, it may draw many people to work on it. So in a sense, it is a kind of highest innovation.
    
  + **If it is not an entirely new problem, why does it still matter?**
    
    > When you pick up a problem to work on, you need to clearly state why it is important.

3. **What is the scientific hypothesis that the paper is trying to verify? ** 

   > Answer to this question is to address what new knowledge is advanced in the paper. A scientific hypothesis sounds like: "If we did abc in our algorithm/dnn architecture, $90 \%$ of case we can guarantee results cde. concrete example, "In ResNet, if we do the Residue block, we expect to be able to learn much deeper networks, which leads to much higher recognition accuracy." 

4. **What are the key related works and who are the key people working on this topic?** 

   > lt is important to identify the most relevant work that inspired the work in this paper. Grouping them helps with a taxonomy helps you to build a systematic view about the research problem addressed. And finally. please memorize the name of the authors and affiliations of these related works, as they will be key people who will appreciate or criticize this work.

5. **What is the key of the proposed solution in the paper?** 

   > Please summarize the key differentiation of the paper when compared with the previous related works.

6. **How are the experiments designed?** 

   > Experiments design is very important. A good experiments design shall validate all claims made in the paper. Indeed, experimetns should be designed around this validation.

7. **What datasets are built/used for the quantitative evaluation?Is the code open sourced?** 

   > Dataset is an important factor in scientific research. And code helps for readers to reproduce the results.

8. **Is the scientific hypothesis well supported by evidence in the experiments?**

   > Are the claims in the paper well supported by the experimental results?

9. **What are the contributions of the paper?** 

   > Up to this point, it should be clear if the paper made one or some solid contributions, which really refers to what knowledge is advanced.

10. **What should/could be done next?** 

   > This shall summarize your understanding of the limitations of the proposed method in the paper. Addressing these limitations are naturally future research, both from the problem definition itself and/or technical improvement. Or it could be linked to some other abstract knowledge in your cognitive model and stimulate new directions to go. This final question is the creativity part.

---

## 非深度方法

### TCA、JDA、BDA

### ARTL

+ label space一致是否可以理解为label的类别一致

+ 什么是核匹配（kernel matching）方法？

+ 流式正则化

+ 如何训练（训练目标为什么能直接作为正则项）

### DMM

+ DMM的训练流程为什么是反的，即先训练了分类器，再去更新特征表示器？
  - 答：文中所描述的只是训练流程的分步，并不代表训练流程。另外，DMM的训练是iterative的，所以先训练分类器，再更新特征提取器是一个理解错误。
+ ***DMM解决了第3个问题吗？（consistent with the discriminative structure）***
  - Discriminative structure指的是数据的判别结构，指的应该是保留数据中用于分类任务的特征。 文中描述关于判别结构的部分应该只有Structural Risk Minimization，但是这是否能保留数据的判别特性，存疑。（我认为类似ARTL中的分类器与特征表示联合训练的方法才会更好的保留判别特征）

### JGSA

+ Data centric method和Subspace centric method的区别？
+ 对于两个域差异过大的问题，本文是如何解决的？

## 深度方法

### DAN

+ MK-MMD
+ 如何用BP算法对MK-MMD中的参数$\beta$进行更新
  - 答：MK-MMD中的参数$\beta$是转化为QP问题后所得到的close form，并不是用BP算法更新的。

### DANN

+ distribution adaptation和feature adaptation的区别
+ 为什么dH(S,T)中不直接用max，而是1-min

### DSN

+ Private Subspace & Shared Subspace
+ Ldiff+Lrecon两个约束来保证private和shared之间的差异性，同时两者结合又能还原原始特征
+ private信息对于最终训练好的Classifier有帮助吗？

