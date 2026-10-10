[English](../README.md) | [简体中文](README.zh-CN.md) | [日本語](README.ja.md) | [한국어](README.ko.md)

# Learn TensorRT：本番環境を見据えた C++ 推論と最適化

[![C++17](https://img.shields.io/badge/C%2B%2B-17-blue.svg)](https://en.cppreference.com/w/cpp/17)
[![TensorRT](https://img.shields.io/badge/TensorRT-10.14.1-green.svg)](https://developer.nvidia.com/tensorrt)
[![CUDA](https://img.shields.io/badge/CUDA-13.0-green.svg)](https://developer.nvidia.com/cuda-toolkit)
[![Container](https://img.shields.io/badge/NGC_PyTorch-25.11-orange.svg)](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/pytorch)

**TensorRT 10.14 による高性能デプロイ**と**低レイテンシパイプライン設計**に焦点を当てた、
実践的な C++17 コースです。**YOLOv8、CUDA 前処理、INT8 量子化、非同期パイプライン、Triton、DeepStream、Nsight プロファイリング**を扱います。

エクスポート済みモデルから計測可能な推論システムまで、完全な YOLOv8 デプロイパイプラインを C++17 で構築します。
RAII による所有権管理、明示的なリソース寿命、ターゲットベースの CMake、防御的な検証、
再現可能なエビデンス、単一の実行ファイルを超えて拡張できるコンポーネントなど、信頼性を支えるエンジニアリング手法を学びます。

性能上重要な経路を最初から最後までたどります。ONNX グラフの検査と修復、動的形状を使用した TensorRT エンジンの構築、
CUDA と NPP による効率的なデータ転送、YOLOv8 の後処理を実装します。その後、明示的 Q/DQ INT8 量子化、
非同期マルチストリームスケジューリング、Sanitizer とソークテスト、Nsight プロファイリングを通じてシステムを最適化・運用します。
高度な選択科目では、プラグイン、Triton、DeepStream、Jetson DLA、OpenVINO、Python バインディング、LLM 推論を扱います。
最終的に得られるのは、単なる `trtexec` コマンドではなく、実行可能で計測可能なデプロイ成果物です。

## 🎬 プロジェクト概要

[![Learn TensorRT のプロジェクト概要を見る](../assets/project-overview-video-thumbnail.jpg)](https://www.youtube.com/watch?v=K1spF7xL_0o)

[YouTube](https://www.youtube.com/watch?v=K1spF7xL_0o) の短いプロジェクト紹介で、
コースの範囲、デプロイワークフロー、PyTorch から本番向け TensorRT 推論に至るエンジニアリングの道筋を確認できます。


## 対象者

このリポジトリは、アルゴリズム研究と本番エンジニアリングの間をつなぎたい開発者を対象としています。

* **デプロイへ移行するアルゴリズムエンジニア：** 深層学習（PyTorch、ONNX）の知識はあるものの、本番品質の推論に必要な C++/TensorRT を習得したい方。
* **初級・中級のデプロイエンジニア：** Python TensorRT API または `trtexec` の使用経験があり、明示的 Q/DQ 量子化、CUDA ゼロコピー、マルチストリーム非同期実行、Nsight プロファイリングへ進みたい方。
* **本番品質のポートフォリオを構築するエンジニア：** シニアレベルのエッジ CV または高性能 AI デプロイ職を目指し、実践的なパイプライン設計を示す参照プロジェクトが必要な方。

**前提知識：** 本コースは本格的な C++ エンジニアリングを扱います。ISO C++17 と CMake の基礎知識が必要です。
Python のみを使用してきた開発者は、まず C++ の所有権、並行処理、CUDA 実行、ビルドシステムの基礎を学んでください。


## 学習ロードマップ

最初に [`docs/learning_roadmap.md`](learning_roadmap.md) を読むことができます。
[`docs/coverage_matrix.md`](coverage_matrix.md) を使うと、コア科目と選択科目を本番デプロイスキルに対応付けられます。
または、コースを順番に進めても構いません。


プロジェクトは小さなレッスン単位で構成されています。各レッスンでは、実行可能な成果物を 1 つと、学んだ内容の短い記録を 1 つ作成します。
共有画像や再利用可能なリソースは、ルートの `assets` フォルダーにあります。

## 利用方法
本コースは Ubuntu 上で開発・テストされています。Windows または WSL でも動作する可能性はありますが、参照環境ではないため追加のトラブルシューティングが必要になる場合があります。

1. 利用可能な NVIDIA GPU を搭載した Ubuntu マシンで、このリポジトリをクローンします。

```bash
git clone git@github.com:Parker-Lyu/Learn-TensorRT.git
cd Learn-TensorRT
```

2. [`00_environment_check/README.md`](../00_environment_check/README.md) に従って、バージョン固定済みの開発環境を起動します。

3. 必要に応じて VS Code と Dev Containers 拡張機能をインストールし、実行中のコンテナへ接続して、マウントされたプロジェクトディレクトリを開きます。

4. コアレッスンを順番に完了し、各チェックポイントでレポートを作成します。その後、すべての選択科目を番号順に進めるのではなく、
目標とする求人内容に合わせて選択科目を選びます。

### コアパス

| ステージ | レッスン | 成果 |
| --- | --- | --- |
| 基礎 | `00_environment_check` から `11_yolov8_trt_cpp` | エンドツーエンドの YOLOv8n TensorRT C++ 推論 |
| チェックポイント 1 | `12_end_to_end_validation_report` | 再現可能な機能・アーキテクチャのエビデンス |
| 最適化 | `13_nsight_performance_diagnosis`、`14_yolov8_int8_quantization_engineering` | プロファイル済みの FP32/FP16/INT8 性能、明示的 Q/DQ 量子化、デプロイのエビデンス |
| チェックポイント 2 | `15_precision_performance_report` | 応募資料として利用できる精度・性能レポート |
| パイプライン | `16_cpp_producer_consumer` から `21_integrated_tensorrt_video_pipeline` | CUDA/NPP 前処理と実際の TensorRT 実行を統合した、有界非同期・マルチストリーム制御 |
| チェックポイント 3 | `22_pipeline_performance_report` | 再現可能な負荷、レイテンシ、スループット、Sanitizer、再起動、30 分間ソークテストのレポート。状態は環境ごとに生成 |


### 選択トラック

レッスン `23`～`31` では、全体で一貫した安定 ID を使用します。番号は、すべての選択科目を数値順に完了すべきことを意味しません。
以下のトラック別の順序と前提条件に従ってください。レッスン `32` が最終統合です。

| トラック | レッスン | 推奨順序と前提条件 |
| --- | --- | --- |
| サーバー推論 | `24_triton_inference_server`、`29_cpp_shared_library_python_binding` | どちらもレッスン 17 を基礎とし、互いに独立しています。 |
| エッジ CV | `27_deepstream_gstreamer_multistream`、`28_jetson_orin_xavier_dla_deployment` | 可能であれば 27 の後に 28 を学習します。28 の完全な検証には Jetson/DLA ハードウェアが必要です。 |
| 高度な TensorRT | `25_onnx_graph_surgery_plugin`、`26_custom_tensorrt_plugin` | **25 → 26**：同じ未対応の `com.acme::AcmeSwish` モデルを、まず GraphSurgeon、次に TensorRT プラグインで修復します。 |
| CPU と Intel | `23_openvino_yolov8` | 独立した分岐です。ドキュメントに従い、レッスン 05/15 の ONNX と比較エビデンスを再利用します。 |
| LLM の基礎理解 | `30_llm_inference_intro` | 独立した分岐で、YOLO/TensorRT の選択科目を前提としません。 |
| CUDA カーネル最適化 | `31_nsight_compute_kernel_analysis` | レッスン 13 を完了し、レッスン 26 の自前カーネル境界を理解してから、小規模 MLP の分析、LayerNorm の最適化、演算子・ネットワーク・本番運用上の判断を行います。 |
| 最終統合 | `32_final_portfolio_case_study` | コアのチェックポイントと、提示する予定の選択科目のエビデンスを完了してから取り組みます。 |


## 学習方法

各レッスンの `README.md` に従ってコードをビルド・実行し、出力を確認します。

期待する出力を得ることは出発点にすぎません。AI を学習支援として利用しながら、
単にプログラムを動かすのではなく、各レッスンのコードを批判的に検討し、実装の理解を深めてください。

## 要件

- 適切な NVIDIA GPU を搭載し、Docker と NVIDIA コンテナを実行できる Ubuntu ホスト。


## 🚀 学習から採用へ

[![キャリア機会](https://img.shields.io/badge/Career-Land_Your_Dream_Role-brightgreen.svg?style=for-the-badge&logo=rocket)](https://github.com/Parker-Lyu/Learn-TensorRT)

このリポジトリが、高性能 AI エンジニアリング分野で理想の職を得る助けになることを願っています。楽しくコーディングしましょう！🎉

## 作者について

Parker Lyu — [LinkedIn](https://www.linkedin.com/in/parker-lyu-ai/)

## ライセンス

このプロジェクトは [MIT License](../LICENSE) の下で提供されています。
