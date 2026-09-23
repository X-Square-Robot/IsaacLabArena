#!/bin/bash
# ================================================================
# IsaacLab Arena 本地一键部署脚本
# 绕过Docker流程，在本地完成完整的Arena环境部署
# 支持 conda 与项目内 uv 环境
# 注意：使用 return 代替 exit，避免在 source 执行时退出终端
# ================================================================

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 打印函数
print_info() { echo -e "${BLUE}[INFO]${NC} $1"; }
print_success() { echo -e "${GREEN}[SUCCESS]${NC} $1"; }
print_warning() { echo -e "${YELLOW}[WARNING]${NC} $1"; }
print_error() { echo -e "${RED}[ERROR]${NC} $1"; }

# 主安装函数（避免使用 exit 退出终端）
_arena_install() {
    # 脚本所在目录（isaaclabarena根目录）
    local ARENA_SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)
    local ARENA_DIR="$ARENA_SCRIPT_DIR"
    # 使用内部的 IsaacLab（与 arena 同级目录）
    local ISAACLAB_DIR="$ARENA_DIR/submodules/IsaacLab"
    local ISAACSIM_PIP_VERSION="6.0.1.0"
    local MANAENV_ROOT
    MANAENV_ROOT=$(cd "$ARENA_DIR/../.." &>/dev/null && pwd)

    # 配置选项
    local ENV_BACKEND="conda"
    local ENV_NAME=""
    local INSTALL_GROOT=false
    local GROOT_DEPS_GROUP="base"
    local BUILD_FLASH_ATTN=false
    local FORCE_REINSTALL=false

    # 帮助信息
    show_help() {
        echo "IsaacLab Arena 本地一键部署脚本"
        echo ""
        echo "用法: source $0 [选项] 或 bash $0 [选项]"
        echo ""
        echo "选项:"
        echo "  --conda <name>    指定 conda 环境名称"
        echo "  --uv <name>       指定项目内 .venv/<name> uv 环境"
        echo "                    - 环境不存在时由 IsaacLab/uv 自动创建 Python 3.12"
        echo "  -g                安装 GR00T 基础依赖"
        echo "  -G <group>        安装 GR00T 指定依赖组 (base, dev, orin, thor, deploy)"
        echo "  --build-flash-attn  从源码构建 GR00T 的 flash-attn"
        echo "  -f                强制重新安装所有依赖"
        echo "  -h                显示此帮助信息"
        echo ""
        echo "示例:"
        echo "  source $0 --conda manaenv      # 使用/创建 manaenv 环境安装"
        echo "  source $0 --uv manaenv         # 使用/创建 ../../.venv/manaenv"
        echo "  source $0 --conda newenv       # 环境不存在时自动执行 stupidSetup 创建"
        echo "  source $0 --conda manaenv -g   # 包含 GR00T 基础依赖"
    }

    # 解析命令行参数
    while [[ $# -gt 0 ]]; do
        case "$1" in
            --conda)
                if [[ -n "${2-}" ]]; then
                    ENV_BACKEND="conda"
                    ENV_NAME="$2"
                    shift 2
                else
                    print_error "--conda 需要一个参数"
                    return 1
                fi
                ;;
            --uv)
                if [[ -n "${2-}" ]]; then
                    ENV_BACKEND="uv"
                    ENV_NAME="$2"
                    shift 2
                else
                    print_error "--uv 需要一个参数"
                    return 1
                fi
                ;;
            -g)
                INSTALL_GROOT=true
                shift
                ;;
            -G)
                if [[ -n "${2-}" ]]; then
                    INSTALL_GROOT=true
                    GROOT_DEPS_GROUP="$2"
                    shift 2
                else
                    print_error "-G 需要一个参数"
                    return 1
                fi
                ;;
            --build-flash-attn)
                BUILD_FLASH_ATTN=true
                shift
                ;;
            -f)
                FORCE_REINSTALL=true
                shift
                ;;
            -h | --help)
                show_help
                return 0
                ;;
            *)
                print_error "无效选项: $1"
                show_help
                return 1
                ;;
        esac
    done

    # 确定最终使用的环境名
    if [ -z "$ENV_NAME" ]; then
        print_error "必须使用 --conda <name> 或 --uv <name> 指定环境"
        show_help
        return 1
    fi
    if [[ ! "$ENV_NAME" =~ ^[A-Za-z0-9._-]+$ ]]; then
        print_error "无效的环境名称: $ENV_NAME"
        return 1
    fi

    local UV_ENV_DIR="$MANAENV_ROOT/.venv/$ENV_NAME"
    local UV_ENV_SPEC="../../../../.venv/$ENV_NAME"

    echo "=========================================="
    echo "   IsaacLab Arena 本地部署脚本"
    echo "=========================================="
    echo ""

    # ================================================================
    # 辅助函数：验证由 ManaEnv Python 拉取器准备的源码
    # ================================================================
    validate_prepared_sources() {
        cd "$ARENA_DIR"

        if [ -f ".gitmodules" ]; then
            local required_sources=(
                "submodules/IsaacLab"
            )
            [ "$INSTALL_GROOT" = true ] && required_sources+=("submodules/Isaac-GR00T")

            local source_path
            for source_path in "${required_sources[@]}"; do
                if [ ! -e "$ARENA_DIR/$source_path/.git" ]; then
                    print_error "缺少已准备的源码: $source_path"
                    print_error "请先由 ManaEnv install.sh 的 Step 1 准备所需源码"
                    return 1
                fi
            done
            print_info "所需源码已准备完成"
        else
            print_warning "未找到 .gitmodules 文件，跳过源码检查"
        fi
    }

    enable_repo_hooks() {
        local HOOK_SETUP_SCRIPT="$ARENA_DIR/script/enable_repo_hooks.sh"

        if [ ! -f "$HOOK_SETUP_SCRIPT" ]; then
            print_warning "未找到 Hook 启用脚本，跳过 Git hook 配置"
            return 0
        fi

        print_info "为 Arena 及子仓库启用 Git hooks..."
        bash "$HOOK_SETUP_SCRIPT" || {
            print_error "Git hook 配置失败"
            return 1
        }
        print_success "Git hooks 配置完成"
    }

    # ================================================================
    # 辅助函数：激活目标环境
    # ================================================================
    activate_target_env() {
        # set +u 下激活：NVIDIA setup_conda_env.sh 裸引用 $PYTHONPATH 等，nounset 下会崩；结束后恢复进入时状态。
        local _restore_u=0
        case $- in *u*) _restore_u=1 ;; esac
        set +u
        # 激活前剥离 `_isaac_sim` 污染；目标环境钩子会重新注入正确路径。
        local _arena_strip_pp _arena_e _arena_ifs="$IFS"
        for _arena_var in PYTHONPATH LD_LIBRARY_PATH; do
            eval "_arena_cur=\${$_arena_var:-}"
            [ -z "$_arena_cur" ] && continue
            _arena_strip_pp=""
            IFS=':'
            for _arena_e in $_arena_cur; do
                case "$_arena_e" in
                    *"/_isaac_sim/"* | *"/_isaac_sim" | "") ;;
                    *) _arena_strip_pp="${_arena_strip_pp:+$_arena_strip_pp:}$_arena_e" ;;
                esac
            done
            IFS="$_arena_ifs"
            if [ -z "$_arena_strip_pp" ]; then unset "$_arena_var"; else export "$_arena_var=$_arena_strip_pp"; fi
        done

        if [ "$ENV_BACKEND" = "uv" ]; then
            if [ ! -f "$UV_ENV_DIR/bin/activate" ]; then
                [ "$_restore_u" = 1 ] && set -u
                print_error "无法激活 uv 环境: $UV_ENV_DIR"
                return 1
            fi
            # shellcheck disable=SC1090
            source "$UV_ENV_DIR/bin/activate"
            [ "$_restore_u" = 1 ] && set -u
            return 0
        fi

        if [ -f "$CONDA_PREFIX/etc/profile.d/conda.sh" ]; then
            source "$CONDA_PREFIX/etc/profile.d/conda.sh"
        elif [ -f "$HOME/miniforge3/etc/profile.d/conda.sh" ]; then
            source "$HOME/miniforge3/etc/profile.d/conda.sh"
        elif [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
            source "$HOME/miniconda3/etc/profile.d/conda.sh"
        elif [ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]; then
            source "$HOME/anaconda3/etc/profile.d/conda.sh"
        elif [ -f "/opt/conda/etc/profile.d/conda.sh" ]; then
            source "/opt/conda/etc/profile.d/conda.sh"
        else
            eval "$(conda shell.bash hook)" 2>/dev/null || true
        fi
        if ! conda activate "$ENV_NAME" 2>/dev/null; then
            [ "$_restore_u" = 1 ] && set -u
            print_error "无法激活 conda 环境: $ENV_NAME"
            return 1
        fi
        [ "$_restore_u" = 1 ] && set -u
        # 显式 return 0：上一行 `[ ... ] && set -u` 在 _restore_u=0 时会返回 1，否则被误判激活失败。
        return 0
    }

    # ================================================================
    # 辅助函数：在剥离 IsaacSim 污染的子进程中运行 conda
    # ================================================================
    # 剥离 `_isaac_sim` 污染后在子进程跑 conda，否则 conda 会崩、拖垮所有 conda 命令。优先复用父级 conda_safe，独立运行时本地剥离。勿用 CONDA_NO_PLUGINS（会废 libmamba）。
    _arena_conda_safe() {
        if declare -F conda_safe >/dev/null 2>&1; then
            conda_safe "$@"
            return $?
        fi
        (
            local clean_py="" clean_ld="" entry IFS_OLD="$IFS"
            IFS=':'
            for entry in ${PYTHONPATH:-}; do
                case "$entry" in
                    *"/_isaac_sim/"* | *"/_isaac_sim" | "") ;;
                    *) clean_py="${clean_py:+$clean_py:}$entry" ;;
                esac
            done
            for entry in ${LD_LIBRARY_PATH:-}; do
                case "$entry" in
                    *"/_isaac_sim/"* | *"/_isaac_sim" | "") ;;
                    *) clean_ld="${clean_ld:+$clean_ld:}$entry" ;;
                esac
            done
            IFS="$IFS_OLD"
            if [ -z "$clean_py" ]; then unset PYTHONPATH; else export PYTHONPATH="$clean_py"; fi
            if [ -z "$clean_ld" ]; then unset LD_LIBRARY_PATH; else export LD_LIBRARY_PATH="$clean_ld"; fi
            conda "$@"
        )
    }

    # 剥离 `_isaac_sim` 污染后在子进程跑 python/pip，否则裸 pip 命中 Isaac 旧 platform.py 崩 `failed to parse CPython sys.version`。优先复用父级 py_safe，独立运行时本地剥离。
    _arena_py_safe() {
        if declare -F py_safe >/dev/null 2>&1; then
            py_safe "$@"
            return $?
        fi
        (
            local clean_py="" clean_ld="" entry IFS_OLD="$IFS"
            IFS=':'
            for entry in ${PYTHONPATH:-}; do
                case "$entry" in
                    *"/_isaac_sim/"* | *"/_isaac_sim" | "") ;;
                    *) clean_py="${clean_py:+$clean_py:}$entry" ;;
                esac
            done
            for entry in ${LD_LIBRARY_PATH:-}; do
                case "$entry" in
                    *"/_isaac_sim/"* | *"/_isaac_sim" | "") ;;
                    *) clean_ld="${clean_ld:+$clean_ld:}$entry" ;;
                esac
            done
            IFS="$IFS_OLD"
            if [ -z "$clean_py" ]; then unset PYTHONPATH; else export PYTHONPATH="$clean_py"; fi
            if [ -z "$clean_ld" ]; then unset LD_LIBRARY_PATH; else export LD_LIBRARY_PATH="$clean_ld"; fi
            "$@"
        )
    }

    _arena_python_executable() {
        if [ "$ENV_BACKEND" = "uv" ]; then
            printf '%s/bin/python\n' "$UV_ENV_DIR"
        else
            command -v python
        fi
    }

    _arena_ensure_uv() {
        local target_python="$1"

        if command -v uv >/dev/null 2>&1; then
            return 0
        fi
        print_info "未检测到 uv，使用目标 Python 引导安装 uv..."
        _arena_py_safe "$target_python" -m pip install uv || {
            print_error "uv 引导安装失败"
            return 1
        }
        command -v uv >/dev/null 2>&1
    }

    _arena_pip() {
        local target_python
        target_python="$(_arena_python_executable)"
        _arena_ensure_uv "$target_python" || return 1
        _arena_py_safe uv pip install --python "$target_python" "$@"
    }

    # ================================================================
    # 第一步：验证环境并安装 IsaacLab
    # ================================================================
    step1_isaaclab_setup() {
        print_info "步骤 1: 设置 IsaacLab 环境"
        print_info "环境后端: $ENV_BACKEND"
        print_info "环境名称: $ENV_NAME"
        print_info "使用 IsaacLab 目录: $ISAACLAB_DIR"

        if [ ! -d "$ISAACLAB_DIR" ]; then
            print_error "找不到 IsaacLab 目录: $ISAACLAB_DIR"
            return 1
        fi

        if [ "$ENV_BACKEND" = "uv" ] && [ -x "$UV_ENV_DIR/bin/python" ]; then
            print_info "检测到已存在的 uv 环境: $UV_ENV_DIR"
        elif [ "$ENV_BACKEND" = "conda" ] && _arena_conda_safe env list | grep -q "^${ENV_NAME} "; then
            print_info "检测到已存在的 conda 环境: $ENV_NAME"
        else
            print_warning "未找到 $ENV_BACKEND 环境: $ENV_NAME"
            print_info "将使用 stupidSetup.sh 创建新环境..."
        fi

        # 无论环境是否存在都交给 stupidSetup.sh 统一处理 IsaacLab 安装：
        # 不存在则创建环境+装 IsaacLab；已存在则由其询问是否重装
        # （-f / ISAACLAB_FORCE_REINSTALL=1 强制重装、跳过询问）。
        if [ ! -f "$ISAACLAB_DIR/stupidSetup.sh" ]; then
            print_error "找不到 stupidSetup.sh: $ISAACLAB_DIR/stupidSetup.sh"
            print_info "请确保 IsaacLab 目录中存在 stupidSetup.sh"
            return 1
        fi

        cd "$ISAACLAB_DIR"
        local env_args=("--$ENV_BACKEND")
        if [ "$ENV_BACKEND" = "uv" ]; then
            env_args+=("$UV_ENV_SPEC")
        else
            env_args+=("$ENV_NAME")
        fi
        local stupid_setup_status=0
        if [ "$FORCE_REINSTALL" = true ]; then
            print_info "执行: ISAACLAB_FORCE_REINSTALL=1 source stupidSetup.sh --$ENV_BACKEND $ENV_NAME（强制重装）"
            ISAACLAB_FORCE_REINSTALL=1 source stupidSetup.sh "${env_args[@]}" || stupid_setup_status=$?
        else
            print_info "执行: source stupidSetup.sh --$ENV_BACKEND $ENV_NAME"
            source stupidSetup.sh "${env_args[@]}" || stupid_setup_status=$?
        fi

        if [ "$stupid_setup_status" -ne 0 ]; then
            print_error "stupidSetup.sh 执行失败"
            return 1
        fi

        print_info "stupidSetup 完成，激活 $ENV_NAME 环境..."
        activate_target_env || return 1

        print_info "验证 Isaac Sim pip 包版本: $ISAACSIM_PIP_VERSION..."
        if ! _arena_py_safe "$(_arena_python_executable)" - "$ISAACSIM_PIP_VERSION" <<'PY'
from importlib.metadata import PackageNotFoundError, version
import sys

try:
    installed = version("isaacsim")
except PackageNotFoundError:
    raise SystemExit("Isaac Sim pip package is not installed")
if installed != sys.argv[1]:
    raise SystemExit(f"expected Isaac Sim {sys.argv[1]}, found {installed}")
print(f"Isaac Sim: {installed}")
PY
        then
            print_error "Isaac Sim pip 包版本验证失败"
            return 1
        fi

        print_info "验证 Python 环境..."
        # _arena_py_safe：裸 python 在污染下会崩，令此检测误报“PyTorch 未检测到”。
        if _arena_py_safe python -c "import torch; print(f'PyTorch: {torch.__version__}')" 2>/dev/null; then
            print_success "PyTorch 已安装"
        else
            print_warning "PyTorch 未检测到"
        fi

        print_success "步骤 1 完成: IsaacLab 环境已就绪"
        echo ""
    }

    # ================================================================
    # 第二步：安装 Arena 核心依赖
    # ================================================================
    step2_install_core_deps() {
        print_info "步骤 2: 安装 Arena 核心依赖"

        cd "$ARENA_DIR"
        activate_target_env || return 1

        print_info "升级 pip..."
        # _arena_py_safe 绕开污染（裸 pip 会崩 `failed to parse CPython sys.version`）+ 检查返回码硬失败（旧版崩了仍 print_success 误报）。
        if ! _arena_pip --upgrade pip; then
            print_error "pip 升级失败"
            return 1
        fi

        print_info "安装核心 Python 依赖..."
        if ! _arena_pip \
            pytest \
            jupyter \
            typing_extensions \
            onnxruntime \
            uv \
            pre-commit; then
            print_error "核心 Python 依赖安装失败"
            return 1
        fi

        print_success "核心依赖安装完成"
        echo ""
    }

    # ================================================================
    # 第三步：安装 Vuer 依赖
    # ================================================================
    step3_install_vuer() {
        print_info "步骤 3: 安装 Vuer 依赖"

        activate_target_env || return 1

        print_info "安装 vuer..."
        # _arena_py_safe：同 step2，绕开激活 Isaac env 后注入的污染。
        if ! _arena_pip vuer; then
            print_error "vuer 安装失败"
            return 1
        fi

        print_success "Vuer 依赖安装完成"
        echo ""
    }

    # ================================================================
    # 第四步：安装 HuggingFace CLI
    # ================================================================
    step4_install_huggingface() {
        print_info "步骤 4: 安装 HuggingFace CLI"

        activate_target_env || return 1

        # _arena_py_safe：同 step2，绕开激活 Isaac env 后注入的污染。
        if ! _arena_pip "huggingface-hub[cli]"; then
            print_error "HuggingFace CLI 安装失败"
            return 1
        fi

        print_success "HuggingFace CLI 安装完成"
        echo ""
    }

    # ================================================================
    # 第五步：安装 OSQP 补丁
    # ================================================================
    step5_install_osqp_patch() {
        print_info "步骤 5: 检查并安装 OSQP 补丁"

        activate_target_env || return 1

        # 检测与安装都走 _arena_py_safe：裸命令在污染下崩，会被误判成“OSQP 缺失”而无谓重装。
        if _arena_py_safe python -c "import qpsolvers; print(qpsolvers.available_solvers)" 2>/dev/null | grep -q "osqp"; then
            print_success "OSQP 已正确安装，无需补丁"
        else
            print_warning "OSQP 缺失，安装补丁版本..."
            if ! _arena_pip qpsolvers==4.8.1; then
                print_error "OSQP 补丁安装失败"
                return 1
            fi
            print_success "OSQP 补丁安装完成"
        fi
        echo ""
    }

    # ================================================================
    # 第六步：可选安装 GR00T 依赖
    # ================================================================
    step6_install_groot() {
        if [ "$INSTALL_GROOT" = false ]; then
            print_info "步骤 6: 跳过 GR00T 安装（使用 -g 或 -G 选项启用）"
            echo ""
            return 0
        fi

        print_info "步骤 6: 安装 GR00T 依赖"
        print_info "依赖组: $GROOT_DEPS_GROUP"

        activate_target_env || return 1

        local GROOT_DIR="$ARENA_DIR/submodules/Isaac-GR00T"

        if [ ! -d "$GROOT_DIR" ]; then
            print_error "找不到 Isaac-GR00T 目录: $GROOT_DIR"
            print_info "请先由 ManaEnv install.sh 的 Step 1 准备所需源码"
            return 1
        fi

        local nvcc_path nvcc_info
        if command -v nvcc >/dev/null 2>&1; then
            nvcc_path="$(command -v nvcc)"
        elif [ -x "/usr/local/cuda-12.8/bin/nvcc" ]; then
            nvcc_path="/usr/local/cuda-12.8/bin/nvcc"
        else
            print_error "未检测到 nvcc；请自行安装 CUDA Toolkit 12.8 后重新运行安装脚本"
            return 1
        fi
        if ! nvcc_info="$("$nvcc_path" --version 2>&1)" || [[ "$nvcc_info" != *"release 12.8"* ]]; then
            print_error "检测到的 CUDA Toolkit 不是 12.8: $nvcc_path"
            print_error "$nvcc_info"
            print_error "请自行安装并配置 CUDA Toolkit 12.8 后重新运行安装脚本"
            return 1
        fi
        print_info "CUDA Toolkit 12.8 已就绪: $nvcc_path"

        local GROOT_DEPS_SCRIPT="$ARENA_DIR/script/install_gr00t_deps_local.sh"
        if [ -f "$GROOT_DEPS_SCRIPT" ]; then
            print_info "执行 GR00T 依赖安装脚本（本地版本）..."
            print_warning "此脚本可能需要 sudo 权限安装系统依赖"
            MANAENV_ENV_BACKEND="$ENV_BACKEND" GROOT_DEPS_GROUP="$GROOT_DEPS_GROUP" ARENA_DIR="$ARENA_DIR" BUILD_FLASH_ATTN="$BUILD_FLASH_ATTN" bash "$GROOT_DEPS_SCRIPT" || return 1
        else
            print_warning "未找到 GR00T 依赖安装脚本"
            print_info "尝试直接安装 Isaac-GR00T..."
            cd "$GROOT_DIR"
            # _arena_py_safe：同上，绕开激活 Isaac env 后注入的污染。
            if ! _arena_pip -e ".[$GROOT_DEPS_GROUP]"; then
                print_error "Isaac-GR00T 安装失败"
                return 1
            fi
        fi

        print_success "GR00T 依赖安装完成"
        echo ""
    }

    # ================================================================
    # 第七步：安装 Arena 模块
    # ================================================================
    step7_install_arena() {
        print_info "步骤 7: 安装 IsaacLab Arena 模块"

        cd "$ARENA_DIR"
        activate_target_env || return 1

        print_info "安装 isaaclab_arena..."
        # _arena_py_safe 绕开污染（裸 pip 会崩）+ 检查返回码硬失败（旧版崩了仍 print_success 误报）。
        if ! _arena_pip -e .; then
            print_error "isaaclab_arena 安装失败 (uv pip install -e . 返回非0)"
            return 1
        fi

        if _arena_py_safe python -c "import isaaclab_arena" 2>/dev/null; then
            print_success "isaaclab_arena 模块安装成功"
        else
            print_error "isaaclab_arena 已通过 uv 安装但 import 失败"
            return 1
        fi

        print_success "Arena 模块安装完成"
        echo ""
    }

    # ================================================================
    # 第八步：配置环境变量和别名
    # ================================================================
    step8_configure_env() {
        print_info "步骤 8: 配置环境变量和别名"

        activate_target_env || return 1

        if [ "$ENV_BACKEND" = "uv" ]; then
            print_info "uv 激活脚本已由 IsaacLab 写入运行时环境变量，无需修改 ~/.bashrc"
        elif grep -q "# IsaacLab Arena Configuration" ~/.bashrc; then
            print_warning "环境配置已存在，跳过"
        else
            cat >>~/.bashrc <<EOF

# IsaacLab Arena Configuration
export ISAACLAB_PATH="\${HOME}/Desktop/new/IsaacLab"

# Arena prompt (for conda env: $ENV_NAME)
if [ "\$CONDA_DEFAULT_ENV" = "$ENV_NAME" ]; then
    PS1='[IsaacLab Arena] \[\e[0;32m\]~\u \[\e[0;34m\]\w\[\e[0m\] \$ '
fi

# Useful aliases
alias ll='ls -alF --color=auto'
alias ..='cd ..'
EOF
            print_info "已添加环境配置到 ~/.bashrc"
        fi

        print_info "安装调试工具 debugpy..."
        # _arena_py_safe 绕开污染；debugpy 非必需，装失败仅告警不阻断。
        _arena_pip debugpy || print_warning "debugpy 安装失败（不影响主流程）"

        print_success "环境配置完成"
        echo ""
    }

    # ================================================================
    # 执行安装流程
    # ================================================================
    print_info "开始 IsaacLab Arena 本地部署..."
    print_info "Arena 目录: $ARENA_DIR"
    print_info "IsaacLab 目录: $ISAACLAB_DIR"
    print_info "环境后端: $ENV_BACKEND"
    print_info "环境名称: $ENV_NAME"
    print_info "安装 GR00T: $INSTALL_GROOT"
    if [ "$INSTALL_GROOT" = true ]; then
        print_info "GR00T 依赖组: $GROOT_DEPS_GROUP"
    fi
    echo ""

    validate_prepared_sources || return 1
    enable_repo_hooks || return 1
    echo ""

    step1_isaaclab_setup || return 1
    step2_install_core_deps || return 1
    step3_install_vuer || return 1
    step4_install_huggingface || return 1
    step5_install_osqp_patch || return 1
    step6_install_groot || return 1
    step7_install_arena || return 1
    step8_configure_env || return 1

    echo "=========================================="
    echo -e "${GREEN}   部署完成!${NC}"
    echo "=========================================="
    echo ""
    print_info "使用方法:"
    if [ "$ENV_BACKEND" = "uv" ]; then
        echo "  1. 激活环境: source $UV_ENV_DIR/bin/activate"
    else
        echo "  1. 激活环境: conda activate $ENV_NAME"
    fi
    echo "  2. 刷新配置: source ~/.bashrc"
    echo "  3. 运行 Arena 示例..."
    echo ""
    print_info "验证安装:"
    echo "  python -c \"import isaaclab_arena; print('Arena OK')\""
    echo ""

    if [ "$INSTALL_GROOT" = true ]; then
        print_info "GR00T 验证:"
        echo "  python -c \"import gr00t; print('GR00T OK')\""
    fi

    return 0
}

# 运行安装函数（传递所有参数）
_arena_install "$@"
