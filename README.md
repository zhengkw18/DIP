# DIP
- HW1: Point Processing
- HW2: Image Warping
- HW3: Content-Aware Image Resizing
- HW4: Image Compression
- Project1: Image Fusion/Face Morphing/View Morphing
- Project2: Non-I.I.D. Image Classification

### 课程总结

十分良心的课，能学到许多传统数字图像处理的知识。崔鹏老师讲课清晰有条理，还不考勤，不听课看看ppt也有收获。

四次小作业两次大作业，这届取消了考试，小作业做两次可以拿满，多做一次总评加5分，得分不要太轻松。

小作业都有往届代码做参考，完全可以混分，不过我还是夹带了一些私货，比如用图形学课渲染的图片做处理、把2D-DCT用矩阵表示了一下。第二次小作业有位同学搞了个叫general surface mapping的东西，我特意去学习了一下，虽然感觉算不上general surface，但那个心形曲面映射确实挺有趣的。

第一次大作业里View Morphing有点难度，其实在Github上搜一下是能找到别人的实现的。三个题目写不出什么新奇的东西，我自己实现了一下Delaunay三角剖分（也是参考了别人的实现），然后渲染了几个视频出来。说实话，那些论文里花哨的数学公式我都不太懂，只是把它们原封不动变成了代码。

第二次大作业因为提不出啥创新，我们小组就是堆工作量，各种论文都实现一下，什么dwr pfdl，网络结构也调俩版本，结果准确率提升微小，只好随便找了一个版本交上预测。展示那天，我们惊喜地得知我们的测试集准确率排11组里的倒数第4。其它组方法也不比我们好啊，只是大多用了个集成学习，感觉亏了一个亿，瞬间对分数没有期望了。令我大受震撼的是准确率前二的做法

1. 添加了一个判别样本是否在训练集的二分类损失，然后用上测试集

2. 搞了一个dropout率高达95%的简单mlp提升泛化能力

那些顶会论文甚至不如这些简单trick，甚至不如简单集成一下，实在是讽刺。

因为小作业加分比较多，虽然预测文件翻车了，最后总评还是拿到了100分（没给A+）。
