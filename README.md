*utils*
**json_generator.py 使用方法 (通过命令行)：**
假设您已将脚本保存为 `utils/json_generator.py`，并且您当前位于项目根目录 (`self_segmentation/`)。

1.  **查看帮助信息**：
    ```bash
    python utils/json_dataset_generator.py --help
    ```
    这将显示所有可用的命令行参数及其描述。

2.  **模式一：生成所有主JSON文件并进行分割 (使用配置中的默认设置)**
    这是最常用的模式，尤其是在项目初始化阶段。
    ```bash
    python utils/json_dataset_generator.py --mode generate_all
    ```
    * **执行流程**：
        1.  脚本会读取 `configs/base.py` 和 `configs/json_config.py` 中的配置（路径、深度、文件名模式等）。
        2.  扫描 `data/raw/` (或配置的原始图像目录) 下的所有图像。
        3.  对于每张原始图像，尝试在 `data/labeled/` (或配置的掩码目录) 下找到同名主体（不同扩展名）的掩码文件。
        4.  将找到的图像-掩码对信息（和未找到掩码的图像信息）以及解析出的序列和帧号，传递给 `build_sequences` 函数。
        5.  `build_sequences` 根据 `INPUT_DEPTH` 构建3D序列样本。
        6.  生成并保存 `master_labeled_dataset.json` 和 `master_unlabeled_dataset.json` 到 `json/` (或配置的JSON输出目录)。
        7.  然后，自动读取刚生成的 `master_labeled_dataset.json`。
        8.  根据配置中的 `DEFAULT_VAL_SPLIT`, `DEFAULT_TEST_SPLIT`, `DEFAULT_RANDOM_SEED` (或命令行覆盖的值) 将其分割为训练、验证、测试集。
        9.  保存 `master_labeled_dataset_train.json`, `master_labeled_dataset_val.json`, `master_labeled_dataset_test.json` 到 `json/` 目录。
    * **示例（覆盖部分配置）**：如果您想临时更改分割比例或文件名模式：
        ```bash
        python utils/json_dataset_generator.py --mode generate_all --val_split 0.2 --test_split 0.1 --pattern "^MySeries_(\d+)\.tif$"
        ```

3.  **模式二：仅生成主 Labeled 和 Unlabeled JSON (不进行分割)**
    如果您只想先看看匹配情况，或者想手动进行更复杂的分割。
    ```bash
    python utils/json_generator.py --mode generate_labeled_unlabeled
    ```
    * **执行流程**：只执行上述 `generate_all` 模式中的步骤 1 到 6。
    * **输出**：`json/master_labeled_dataset.json` 和 `json/master_unlabeled_dataset.json`。

4.  **模式三：对一个已存在的 Labeled JSON 文件进行分割**
    如果您已经有了一个包含所有已标注样本的 `master_labeled_dataset.json`，并希望重新分割它或者用不同的比例分割。
    ```bash
    python utils/json_generator.py --mode split_labeled --input_json master_labeled_dataset.json --val_split 0.1 --test_split 0.1 --seed 123
    ```
    * **执行流程**：
        1.  脚本会在 `configs/json_config.py` 中配置的 `JSON_OUTPUT_DIR_NAME` (例如 `json/`) 目录下查找名为 `master_labeled_dataset.json` (由 `--input_json` 指定) 的文件。
        2.  加载该JSON文件中的样本列表。
        3.  根据命令行提供的（或配置中的）分割比例和随机种子进行分割。
        4.  保存 `master_labeled_dataset_train.json`, `master_labeled_dataset_val.json`, `master_labeled_dataset_test.json`。
    * **注意**：`--input_json` 参数只需要提供**文件名**，脚本会在配置的JSON输出目录中查找它。
