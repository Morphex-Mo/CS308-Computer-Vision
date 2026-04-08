# CS308 Assignment 1 报告

## Hybrid Image 示例

我使用固定的 Gaussian cutoff（cutoff = 7）生成了 5 组主实验 hybrid image。

- bird + plane：结果偏向边缘细节，缩小后飞机轮廓更明显。
- submarine + fish：混合效果最自然，近看能看到鱼的纹理，远看整体结构也比较连贯。
- einstein + marilyn：不同尺度下的人脸特征切换明显，符合 hybrid image 的典型效果。
- motorcycle + bicycle：线条和轮辐细节较丰富，多尺度感比较清楚。
- cat + dog：近看时能看到狗的五官与毛发细节，远看时猫的整体轮廓更容易被感知。

## Bonus Step 1：arknights + honkai（固定 cutoff = 7）

我将 arknights 与 honkai 按与 cat/dog 相同流程进行混合，并输出到 results/bonus/my_hybrid_example。

- 该组合能够形成有效 hybrid image。
- 由于两张图都具有较高纹理密度和复杂色彩，远近主导信息切换不如人脸或猫狗样例直观。

## Bonus Step 2：三组图片的 cutoff 参数调节（1~8）

调参组合为：cat_dog、submarine_fish、bird_plane。

- cat_dog：
	- cutoff=1 时图像较干净，但“近看狗”高频感偏弱。
	- cutoff=4 时狗的细节开始明显，远近切换更自然。
	- cutoff=8 时轮廓叠加更强，重影感更明显。
	- 推荐区间约为 4~6。

- submarine_fish：
	- cutoff=1 时潜艇轮廓占主导，鱼纹理偏弱。
	- cutoff=4 时鱼体细节与背景结构平衡最好。
	- cutoff=8 时细节增强，但混合感增加、边界更复杂。
	- 推荐值约为 4（可扩展到 4~6）。

- bird_plane：
	- cutoff=1 时鸟的高频边缘较强，飞机主体不够稳定。
	- cutoff=4 时飞机结构更清楚，结果更平衡。
	- cutoff=8 与 cutoff=4 接近，但整体更平滑，冲突略减。
	- 推荐区间为 4~8；偏稳定可取 6~8，偏切换感可取 4~6。

## 结论

综合来看，cutoff 主要控制“高频细节强度”和“重影风险”之间的平衡。

- cutoff 偏小：结果更干净，但切换感偏弱。
- cutoff 偏大：高频叠加更强，但更容易出现混合冲突。
- 在本次数据中，中等区间（约 4~6）整体最稳健，通常能同时保留可识别细节与较自然的远近切换。
