# CS308 Assignment 1 报告

## Hybrid Image 示例

我使用固定的 Gaussian cutoff（`cutoff = 7`）生成了 5 组 hybrid image 示例。

- `bird + plane`：结果偏向边缘细节，缩小后飞机轮廓更明显。
- `submarine + fish`：混合效果最自然，近看能看到鱼的纹理，远看整体结构也比较连贯。
- `einstein + marilyn`：不同尺度下的人脸特征切换明显，符合 hybrid image 的典型效果。
- `motorcycle + bicycle`：线条和轮辐细节较丰富，多尺度感比较清楚。
- `arknights + honkai`：画面内容较复杂，但在不同尺度下仍能看出切换效果。

## Extra credit : cutoff 参数分析

我在 extra credit 中测试了 `1` 到 `8` 的 cutoff。

- 较小的 cutoff（`1~2`）会保留更强的高频细节，但边缘冲突和伪影也更明显。
- 中等 cutoff（`3~6`）通常能在低频结构和高频细节之间取得最好平衡。
- 较大的 cutoff（`7~8`）会让图像更平滑、更稳定，但 hybrid 效果会减弱，因为高频内容被削弱得更多。

## 结论

综合来看，我的结果在 `cutoff = 4~6` 左右视觉效果最好。这个范围既能保留清晰的多尺度变化，又不会让图像过于失真。
