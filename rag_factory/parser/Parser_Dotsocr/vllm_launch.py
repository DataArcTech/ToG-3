import os
import subprocess
import sys
from pathlib import Path
import argparse

def launch_vllm_server(hf_model_path="weights/DotsOCR", num_gpus="0", gpu_memory_utilization=0.95, port=8001):
    # 1. 检查模型路径
    model_path = Path(hf_model_path).resolve()
    if not model_path.exists():
        print(f"error: 模型路径不存在: {model_path}")
        sys.exit(1)

    # 2. 设置环境变量
    os.environ["hf_model_path"] = str(model_path)
    os.environ["PYTHONPATH"] = f"{model_path.parent}:{os.environ.get('PYTHONPATH', '')}" # 
    os.environ["CUDA_VISIBLE_DEVICES"] = num_gpus
    os.environ["OPENAI_API_BASE"] = f"http://localhost:{port}/v1"
    os.environ["OPENAI_API_KEY"] = "EMPTY"

    # 3. 修改 vllm CLI 添加模型 
    try:
        vllm_path = subprocess.check_output(["which", "vllm"], text=True).strip()
        with open(vllm_path, "r") as f:
            vllm_content = f.read()

        inject_line = "from DotsOCR import modeling_dots_ocr_vllm"
        if inject_line not in vllm_content:
            print("修改 vllm CLI 引入 DotsOCR 模型...")
            sed_cmd = f"sed -i '/^from vllm\\.entrypoints\\.cli\\.main import main$/a\\{inject_line}' {vllm_path}"
            subprocess.run(sed_cmd, shell=True, check=True)
        else:
            print("vllm CLI 已包含 DotsOCR 模型")

    except subprocess.CalledProcessError as e:
        print(f"error: 获取 vllm 路径失败: {e}")
        sys.exit(1)

    # 4. 启动 vllm server
    print(" 正在启动 vLLM 服务...")
    cmd = [
        "vllm", "serve", str(model_path),
        "--tensor-parallel-size", "1",
        "--gpu-memory-utilization", str(gpu_memory_utilization),
        "--chat-template-content-format", "string",
        "--served-model-name", "model",
        "--trust-remote-code",
        "--port", str(port)
    ]

    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"error: vLLM 启动失败: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Launch vLLM server with dots_ocr model.")
    parser.add_argument("--model_path", type=str, required=True, help="Path to your downloaded model weights, Please use a directory name without periods")
    parser.add_argument("--gpus", type=str, default="0", help="GPU device ID(s)")
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.95, help="Desired GPU memory utilization percentage")
    parser.add_argument("--port", type=int, default=8001, help="Port to launch the vLLM server on")

    args = parser.parse_args()

    launch_vllm_server(args.model_path, args.gpus, args.gpu_memory_utilization, args.port)
