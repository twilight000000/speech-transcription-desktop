from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any
import streamlit as st
import torch
from model_resources import diarization_available

from 控制台 import (
    DEFAULT_INPUT_DIR,
    DEFAULT_OUTPUT_DIR,
    PERSISTED_SETTING_KEYS,
    SCRIPT_PATH,
    TaskProgress,
    build_command,
    build_language_code_options,
    extract_effective_config_from_cmd,
    get_task_start_blocker,
    kill_tracked_task,
    language_label,
    list_pending_task_interactions,
    load_hf_token,
    load_last_diarization_abort,
    load_task_meta,
    load_task_result,
    load_user_settings,
    parse_cli_tokens,
    parse_progress,
    read_log_tail,
    recommend_device_mode,
    respond_to_task_interaction,
    save_hf_token,
    save_user_settings,
    save_task_meta,
    set_tracked_task_paused,
    start_task,
    tracked_task_state,
)


def apply_ui_tweaks() -> None:
    # 隐藏 Streamlit 右上角工具栏，并压缩布局留白。
    st.markdown(
        """
<style>
div[data-testid="stToolbar"] {visibility: hidden; height: 0; position: fixed;}
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}

.block-container {padding-top: 1rem; padding-bottom: 0.8rem; max-width: 96%;}
div[data-testid="stVerticalBlock"] {gap: 0.45rem;}
div[data-testid="stMetric"] {
    background: #f7f8fa;
    border: 1px solid #e8ebef;
    border-radius: 8px;
    padding: 0.45rem 0.65rem;
}
</style>
""",
        unsafe_allow_html=True,
    )


def init_session_state() -> None:
    remembered_hf_token = os.getenv("HF_TOKEN", "").strip() or load_hf_token()
    defaults: dict[str, Any] = {
        "task_input_dir": DEFAULT_INPUT_DIR,
        "task_output_dir": DEFAULT_OUTPUT_DIR,
        "task_language_search": "",
        "task_language_code": "",
        "task_precision_mode": "均衡",
        "task_device_mode": "仅GPU" if torch.cuda.is_available() else "仅CPU",
        "task_utilization": "均衡",
        "task_heartbeat_sec": 15,
        "task_enable_diarization": False,
        "task_parallel_workers": 0,
        "task_diarization_unavailable_policy": "运行时询问",
        "task_resume_from_file": "",
        "task_hf_token": remembered_hf_token,
        "task_write_json": True,
        "task_skip_existing": True,
        "task_auto_env_tune": False,
        "task_maximize_utilization": False,
        "ui_auto_refresh": True,
        "ui_refresh_interval_sec": 2,
        "ui_log_tail_lines": 500,
        "ui_log_filter_keyword": "",
        "synced_running_pid": 0,
    }
    defaults.update(load_user_settings())
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def collect_session_user_settings() -> dict[str, Any]:
    return {key: st.session_state[key] for key in PERSISTED_SETTING_KEYS if key in st.session_state}


def save_session_user_settings() -> None:
    try:
        save_user_settings(collect_session_user_settings())
    except Exception:
        pass


def sync_task_fields_from_cmd(task_meta: dict[str, Any]) -> None:
    pid = int(task_meta.get("pid", 0))
    if pid <= 0:
        return
    if st.session_state.get("synced_running_pid", 0) == pid:
        return
    cmd = task_meta.get("cmd", [])
    if not isinstance(cmd, list):
        return

    values, flags = parse_cli_tokens([str(x) for x in cmd])
    st.session_state["task_input_dir"] = values.get("--input-dir", st.session_state["task_input_dir"])
    st.session_state["task_output_dir"] = values.get("--output-dir", st.session_state["task_output_dir"])
    st.session_state["task_language_code"] = values.get("--language", "")
    st.session_state["task_resume_from_file"] = values.get("--resume-from-file", "")
    if "--parallel-workers" in values:
        try:
            st.session_state["task_parallel_workers"] = max(0, int(values["--parallel-workers"]))
        except Exception:
            st.session_state["task_parallel_workers"] = 0
    st.session_state["task_device_mode"] = {
        "cpu": "仅CPU",
        "cuda": "仅GPU",
        "auto": "CPU+GPU协同" if "--hybrid" in flags else "自动",
    }.get(values.get("--device", "auto"), "自动")

    speed_profile = values.get("--speed-profile", "balanced")
    st.session_state["task_precision_mode"] = {
        "accurate": "高精度",
        "fast": "高速",
    }.get(speed_profile, "均衡")

    # 无法稳定逆推占用档位时，保留现有用户选择。
    st.session_state["task_enable_diarization"] = "--disable-diarization" not in flags
    st.session_state["task_diarization_unavailable_policy"] = {
        "ask": "运行时询问",
        "transcribe-only": "自动仅转写",
        "abort": "立即中断任务",
    }.get(
        values.get("--diarization-unavailable-policy", "ask"),
        st.session_state["task_diarization_unavailable_policy"],
    )
    st.session_state["task_write_json"] = "--write-json" in flags
    st.session_state["task_skip_existing"] = "--skip-existing" in flags
    st.session_state["task_auto_env_tune"] = "--auto-env-tune" in flags and "--disable-auto-env-tune" not in flags
    st.session_state["task_maximize_utilization"] = "--maximize-utilization" in flags
    if "--heartbeat-sec" in values:
        try:
            st.session_state["task_heartbeat_sec"] = max(2, int(values["--heartbeat-sec"]))
        except Exception:
            st.session_state["task_heartbeat_sec"] = 15
    st.session_state["synced_running_pid"] = pid


def show_hardware_info() -> None:
    cpu_threads = os.cpu_count() or 0
    cuda_available = torch.cuda.is_available()

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("CPU 线程", str(cpu_threads))
    col2.metric("CUDA 可用", "是" if cuda_available else "否")
    col3.metric("GPU 数量", str(torch.cuda.device_count() if cuda_available else 0))

    if cuda_available:
        try:
            gpu_mem_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        except Exception:
            gpu_mem_gb = 0.0
        col4.metric("首张 GPU 显存", f"{gpu_mem_gb:.2f}GB" if gpu_mem_gb else "-")
        st.info(f"检测到 GPU: {torch.cuda.get_device_name(0)}")
    else:
        col4.metric("首张 GPU 显存", "-")
        st.warning("未检测到可用 CUDA；选择 GPU/协同时，自动调节关闭会报错，开启后才允许程序自动回退。")


def render_dashboard(
    task_meta: dict[str, Any] | None,
    task_running: bool,
    task_paused: bool,
    progress: dict[str, Any],
) -> None:
    st.subheader("任务仪表盘")
    pid = int(task_meta.get("pid", 0)) if task_meta else 0
    total_files = int(progress.get("total_files", 0))
    finished_files = int(progress.get("finished_files", 0))
    fail_count = int(progress.get("fail_count", 0))

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("任务 PID", str(pid) if pid else "-")
    if task_running and task_paused:
        status_text = "已暂停"
    elif task_running:
        status_text = "运行中"
    else:
        status_text = "空闲/已结束"
    col2.metric("运行状态", status_text)
    if total_files > 0:
        col3.metric("完成进度", f"{finished_files}/{total_files}")
        ratio = min(1.0, finished_files / total_files)
        st.progress(ratio, text=f"总体进度：{finished_files}/{total_files}")
    else:
        col3.metric("完成进度", f"{finished_files}/?")
        st.progress(0.0, text="等待日志产出任务总量...")
    col4.metric("失败数量", str(fail_count))

    if task_running and task_paused:
        st.warning("任务已暂停；点击“继续任务”可恢复执行。任务参数仍保持锁定。")
        st.info(f"暂停前阶段：{progress.get('stage', '等待启动')}")
        paused_step = str(progress.get("step", "")).strip()
        if paused_step:
            st.info(f"暂停前步骤：{paused_step}")
    elif task_running:
        st.warning("当前任务参数已锁定；仅“显示与日志设置”中的选项可实时调整。")
        st.info(f"当前阶段：{progress.get('stage', '等待启动')}")
        current_step = str(progress.get("step", "")).strip()
        if current_step:
            st.info(f"当前步骤：{current_step}")
    elif task_meta:
        result = load_task_result(task_meta)
        result_status = str((result or {}).get("status", "")).strip()
        result_note = str((result or {}).get("note", "")).strip()
        result_failed = (result or {}).get("failed", [])
        if bool(task_meta.get("stop_requested", False)) and not result:
            st.warning("任务已由用户停止。")
        elif result_status == "ok":
            st.success("任务已完成（后端结果状态正常）。")
        elif result_status == "partial-failed":
            result_failed_count = len(result_failed) if isinstance(result_failed, list) else fail_count
            st.warning(f"任务已结束，失败文件数：{result_failed_count}。")
        elif result_status in {"exited", "crashed"}:
            st.error(result_note or "后端任务异常退出，未提供具体原因。")
        elif result_status == "aborted":
            st.warning(result_note or "任务已中断。")
        elif progress.get("done_flag") and progress.get("success_flag"):
            st.success("任务已完成（日志显示全部成功）。")
        elif progress.get("done_flag"):
            st.warning(f"任务已结束（日志显示有失败，失败数：{fail_count}）。")
        else:
            st.warning("任务进程已结束，但日志未写入完整收尾信息，可能是手动停止。")


def main() -> None:
    st.set_page_config(
        page_title="语音识别算力控制台",
        layout="wide",
        menu_items={"About": "语音识别算力控制台"},
    )
    apply_ui_tweaks()
    init_session_state()

    st.title("语音识别算力控制台")
    st.caption("任务参数在运行中自动锁定，页面刷新后仍可追踪真实进程。")

    if not SCRIPT_PATH.exists():
        st.error(f"未找到脚本: {SCRIPT_PATH}")
        return

    task_meta = load_task_meta()
    state = "stopped"
    task_running = False
    task_paused = False
    if task_meta and "pid" in task_meta:
        state = tracked_task_state(task_meta)
        task_running = state != "stopped"
        if state == "unknown":
            st.warning("暂时无法核实任务进程身份，暂停、继续和停止操作已锁定。")
        if task_running:
            sync_task_fields_from_cmd(task_meta)
            task_paused = bool(task_meta.get("paused", False))
        elif task_meta.get("paused"):
            task_meta["paused"] = False
            save_task_meta(task_meta)

    if "_progress_reader" not in st.session_state:
        st.session_state["_progress_reader"] = TaskProgress()
    progress = (st.session_state["_progress_reader"].read(Path(task_meta["log_path"]))
                if task_meta and task_meta.get("log_path") else parse_progress([]))

    render_dashboard(task_meta, task_running, task_paused, progress)

    pending_interactions = list_pending_task_interactions(task_meta) if task_running else []
    if pending_interactions:
        request = pending_interactions[0]
        request_id = str(request.get("request_id", ""))
        summary = str(request.get("summary", "")).strip() or "当前文件无法执行说话人分离。"
        st.warning(summary)
        st.info("请在主窗口选择当前文件是“仅转写继续”还是“中断任务”。")
        interaction_col1, interaction_col2 = st.columns([1, 1])
        if interaction_col1.button(
            "当前文件仅转写并继续",
            type="primary",
            key=f"interaction_continue_{request_id}",
        ):
            ok, message = respond_to_task_interaction(task_meta, request_id, "transcribe-only")
            if not ok:
                st.error(message)
            else:
                st.rerun()
        if interaction_col2.button(
            "立即中断任务",
            key=f"interaction_abort_{request_id}",
        ):
            ok, message = respond_to_task_interaction(task_meta, request_id, "abort")
            if not ok:
                st.error(message)
            else:
                st.rerun()

    button_col1, button_col2, button_col3, button_col4 = st.columns([1, 1, 1, 1])
    start_clicked = button_col1.button("开始任务", type="primary", disabled=task_running)
    pause_clicked = button_col2.button("暂停任务", disabled=state != "running" or task_paused)
    resume_clicked = button_col3.button("继续任务", disabled=state != "running" or not task_paused)
    stop_clicked = button_col4.button("停止任务", disabled=state != "running")

    with st.expander("显示与日志设置（运行中可实时生效）", expanded=True):
        st.session_state["ui_auto_refresh"] = st.toggle(
            "运行中自动刷新",
            value=bool(st.session_state["ui_auto_refresh"]),
        )
        refresh_candidates = [1, 2, 5]
        if st.session_state["ui_refresh_interval_sec"] not in refresh_candidates:
            st.session_state["ui_refresh_interval_sec"] = 2
        st.session_state["ui_refresh_interval_sec"] = st.selectbox(
            "刷新间隔（秒）",
            options=refresh_candidates,
            index=refresh_candidates.index(int(st.session_state["ui_refresh_interval_sec"])),
        )
        tail_candidates = [200, 500, 1000]
        if st.session_state["ui_log_tail_lines"] not in tail_candidates:
            st.session_state["ui_log_tail_lines"] = 500
        st.session_state["ui_log_tail_lines"] = st.selectbox(
            "日志显示行数",
            options=tail_candidates,
            index=tail_candidates.index(int(st.session_state["ui_log_tail_lines"])),
        )
        st.session_state["ui_log_filter_keyword"] = st.text_input(
            "日志过滤关键字",
            value=str(st.session_state["ui_log_filter_keyword"]),
            help="支持中文或代码关键字过滤日志显示。",
        )

    params_locked = task_running
    last_abort_meta = load_last_diarization_abort()
    with st.expander("上次中断恢复点", expanded=bool(last_abort_meta)):
        if not last_abort_meta:
            st.info("暂无已记录的中断恢复点。")
        else:
            st.write(f"时间：{last_abort_meta.get('timestamp', '-')}")
            st.write(f"文件：{last_abort_meta.get('resume_from_file', '-')}")
            st.write(f"阶段：{last_abort_meta.get('stage', '-')}")
            st.write(f"原因：{last_abort_meta.get('reason', '-')}")
            if st.button("加载上次中断文件为恢复起点", disabled=params_locked):
                st.session_state["task_input_dir"] = str(
                    last_abort_meta.get("input_dir", st.session_state["task_input_dir"])
                )
                st.session_state["task_output_dir"] = str(
                    last_abort_meta.get("output_dir", st.session_state["task_output_dir"])
                )
                st.session_state["task_resume_from_file"] = str(
                    last_abort_meta.get("resume_from_file", "")
                )
                st.rerun()

    with st.expander("任务配置（运行中锁定）", expanded=True):
        with st.expander("当前硬件状态", expanded=True):
            show_hardware_info()

        col_left, col_right = st.columns([1, 1])
        with col_left:
            st.session_state["task_input_dir"] = st.text_input(
                "输入音视频目录",
                value=st.session_state["task_input_dir"],
                disabled=params_locked,
            )
            st.session_state["task_output_dir"] = st.text_input(
                "输出目录",
                value=st.session_state["task_output_dir"],
                disabled=params_locked,
            )
            st.session_state["task_resume_from_file"] = st.text_input(
                "从指定文件恢复",
                value=str(st.session_state["task_resume_from_file"]),
                disabled=params_locked,
                help="留空表示从头开始。可填写相对输入目录的路径，例如 子目录/001.wav。",
            )
            st.session_state["task_language_search"] = st.text_input(
                "搜索语言（中文名/代码）",
                value=st.session_state["task_language_search"],
                disabled=params_locked,
            )
            language_options = build_language_code_options(
                search_keyword=str(st.session_state["task_language_search"]),
                current_code=str(st.session_state["task_language_code"]),
            )
            current_code = str(st.session_state["task_language_code"])
            if current_code not in language_options:
                current_code = ""
                st.session_state["task_language_code"] = ""
            language_index = language_options.index(current_code) if current_code in language_options else 0
            st.session_state["task_language_code"] = st.selectbox(
                "识别语言",
                options=language_options,
                index=language_index,
                format_func=language_label,
                disabled=params_locked,
            )
            st.session_state["task_precision_mode"] = st.selectbox(
                "识别精度",
                ["均衡", "高精度", "高速"],
                index=["均衡", "高精度", "高速"].index(str(st.session_state["task_precision_mode"])),
                disabled=params_locked,
            )
            st.session_state["task_device_mode"] = st.selectbox(
                "硬件模式",
                ["自动", "仅CPU", "仅GPU", "CPU+GPU协同"],
                index=["自动", "仅CPU", "仅GPU", "CPU+GPU协同"].index(str(st.session_state["task_device_mode"])),
                disabled=params_locked,
            )
            parallel_candidates = [0, 1, 2, 3, 4]
            if int(st.session_state["task_parallel_workers"]) not in parallel_candidates:
                st.session_state["task_parallel_workers"] = 0
            st.session_state["task_parallel_workers"] = st.selectbox(
                "文件级并行 worker",
                options=parallel_candidates,
                index=parallel_candidates.index(int(st.session_state["task_parallel_workers"])),
                format_func=lambda x: "自动" if int(x) == 0 else str(x),
                disabled=params_locked,
                help="多 worker 主要用于 CPU 批处理；GPU/协同模式下显式选择多个 worker 会在启动时提示错误。",
            )
            st.session_state["task_utilization"] = st.select_slider(
                "运算占用",
                options=["低占用", "均衡", "性能拉满"],
                value=str(st.session_state["task_utilization"]),
                disabled=params_locked,
            )
            heartbeat_candidates = [5, 10, 15, 20, 30]
            if int(st.session_state["task_heartbeat_sec"]) not in heartbeat_candidates:
                st.session_state["task_heartbeat_sec"] = 15
            st.session_state["task_heartbeat_sec"] = st.selectbox(
                "心跳日志间隔（秒）",
                options=heartbeat_candidates,
                index=heartbeat_candidates.index(int(st.session_state["task_heartbeat_sec"])),
                disabled=params_locked,
            )

        with col_right:
            st.session_state["task_auto_env_tune"] = st.toggle(
                "程序自动调节参数",
                value=bool(st.session_state["task_auto_env_tune"]),
                disabled=params_locked,
                help="默认关闭。开启后才允许程序自动修正不兼容参数；关闭时只报错不修改你的设置。",
            )
            st.session_state["task_maximize_utilization"] = st.toggle(
                "高占用优化（保精度）",
                value=bool(st.session_state["task_maximize_utilization"]),
                disabled=params_locked,
                help="用户显式开启后才会提升线程/batch/chunk；关闭时不会自动拉高占用。",
            )
            st.session_state["task_enable_diarization"] = st.toggle(
                "启用说话人分离",
                value=bool(st.session_state["task_enable_diarization"]),
                disabled=params_locked,
            )
            st.session_state["task_diarization_unavailable_policy"] = st.selectbox(
                "说话人分离不可用时",
                ["运行时询问", "自动仅转写", "立即中断任务"],
                index=["运行时询问", "自动仅转写", "立即中断任务"].index(
                    str(st.session_state["task_diarization_unavailable_policy"])
                ),
                disabled=params_locked,
                help="遇到某个文件无法做说话人分离时，决定是当场询问、自动退化为仅转写，还是直接中断任务。",
            )
            st.session_state["task_hf_token"] = st.text_input(
                "Hugging Face Token（仅在线下载说话人模型时需要）",
                value=str(st.session_state["task_hf_token"]),
                type="password",
                disabled=params_locked,
            )
            st.session_state["task_write_json"] = st.toggle(
                "输出结构化 JSON",
                value=bool(st.session_state["task_write_json"]),
                disabled=params_locked,
            )
            st.session_state["task_skip_existing"] = st.toggle(
                "跳过已处理文件",
                value=bool(st.session_state["task_skip_existing"]),
                disabled=params_locked,
            )
            st.markdown("**占用档位说明**")
            st.write("- 低占用：减少线程和 batch，适合后台慢跑")
            st.write("- 均衡：默认推荐")
            st.write("- 性能拉满：尽可能吃满硬件")
            st.write("- 识别精度：高精度更准更慢，高速更快但可能降准")
            st.write("- 协同模式：同一文件内，GPU 负责转写，CPU 负责说话人分离")
            st.write("- 文件级并行：按文件切分给多个 worker；多个 worker 主要用于 CPU 批处理")
            st.write("- 程序自动调节参数：默认关闭；开启后才会自动修正不兼容参数")
            st.write("- 高占用优化：默认关闭；显式开启后会在保精度前提下提高占用")
            st.write("- 心跳日志：长步骤每 N 秒输出一次进度存活信息")
            st.write("- chunk 建议不超过 30 秒；关闭自动调节时，超出范围会报错")
            st.write("- 运行时询问：会在主窗口要求你选择“仅转写继续”或“立即中断”")

        recommended_mode, recommendation_text = recommend_device_mode(
            str(st.session_state["task_input_dir"]),
            bool(st.session_state["task_enable_diarization"]),
        )
        st.info(
            "模式声明：单个音频优先单 GPU；多文件批处理且启用说话人分离时，再考虑 CPU+GPU 协同。"
            f" 当前推荐：{recommended_mode}。{recommendation_text}"
        )
        if not params_locked and str(st.session_state["task_device_mode"]) != recommended_mode:
            if st.button("应用当前推荐硬件模式"):
                st.session_state["task_device_mode"] = recommended_mode
                st.rerun()
        if (
            int(st.session_state["task_parallel_workers"]) > 1
            and str(st.session_state["task_device_mode"]) in {"仅GPU", "CPU+GPU协同"}
        ):
            st.warning("当前硬件模式下不支持多个文件级 worker；程序自动调节参数关闭时，启动会报错而不会自动回退为 1。")

        cmd = build_command(
            input_dir=str(st.session_state["task_input_dir"]),
            output_dir=str(st.session_state["task_output_dir"]),
            language=str(st.session_state["task_language_code"]),
            precision_mode=str(st.session_state["task_precision_mode"]),
            utilization=str(st.session_state["task_utilization"]),
            device_mode=str(st.session_state["task_device_mode"]),
            parallel_workers=int(st.session_state["task_parallel_workers"]),
            resume_from_file=str(st.session_state["task_resume_from_file"]),
            enable_diarization=bool(st.session_state["task_enable_diarization"]),
            diarization_unavailable_policy=str(st.session_state["task_diarization_unavailable_policy"]),
            hf_token=str(st.session_state["task_hf_token"]),
            write_json=bool(st.session_state["task_write_json"]),
            skip_existing=bool(st.session_state["task_skip_existing"]),
            auto_env_tune=bool(st.session_state["task_auto_env_tune"]),
            maximize_utilization=bool(st.session_state["task_maximize_utilization"]),
            heartbeat_sec=int(st.session_state["task_heartbeat_sec"]),
        )

        st.markdown("**执行命令预览**")
        st.code(" ".join(f'"{c}"' if " " in c else c for c in cmd), language="bash")
        if str(st.session_state["task_resume_from_file"]).strip():
            st.info(
                "本次任务将从指定文件开始处理："
                + str(st.session_state["task_resume_from_file"]).strip()
            )
        if (
            bool(st.session_state["task_maximize_utilization"])
            and str(st.session_state["task_utilization"]) == "低占用"
        ):
            st.info("当前选择低占用，已优先采用低占用参数，不会追加高占用优化。")
        if (
            bool(st.session_state["task_maximize_utilization"])
            and str(st.session_state["task_utilization"]) != "低占用"
            and str(st.session_state["task_device_mode"]) == "CPU+GPU协同"
        ):
            st.info("已开启高占用优化：本次将使用“协同高占用”（GPU转写 + CPU分离并行增强）。")
        if bool(st.session_state["task_enable_diarization"]) and not diarization_available(str(st.session_state["task_hf_token"])):
            if bool(st.session_state["task_auto_env_tune"]):
                st.warning("你启用了说话人分离但未填写 HF_TOKEN；已开启程序自动调节参数时，运行时会降级为仅转写。")
            else:
                st.warning("你启用了说话人分离但未填写 HF_TOKEN；程序自动调节参数关闭时，启动会报错。")
        elif (
            bool(st.session_state["task_enable_diarization"])
            and str(st.session_state["task_diarization_unavailable_policy"]) == "运行时询问"
        ):
            st.info("若某个文件无法做说话人分离，主窗口会显示“仅转写继续”或“中断任务”的选择。")

    save_session_user_settings()

    if start_clicked:
        if task_running:
            st.warning("已有任务在运行，请先停止当前任务。")
        else:
            hf_token = str(st.session_state["task_hf_token"])
            blocker = get_task_start_blocker(cmd, hf_token)
            if blocker:
                st.error(blocker)
            else:
                save_session_user_settings()
                try:
                    save_hf_token(hf_token)
                except Exception as exc:
                    st.warning(f"Token 可用于本次任务，但本机加密保存失败：{exc}")
                try:
                    task_meta = start_task(cmd, hf_token)
                except Exception as exc:
                    st.error(f"启动任务失败：{exc}")
                else:
                    st.success(f"任务已启动，PID={task_meta['pid']}")
                    st.rerun()

    if pause_clicked and task_meta:
        ok, message = set_tracked_task_paused(task_meta, pause=True)
        if ok:
            task_meta["paused"] = True
            save_task_meta(task_meta)
            st.warning(message)
        else:
            st.error(f"暂停失败: {message}")
        st.rerun()

    if resume_clicked and task_meta:
        ok, message = set_tracked_task_paused(task_meta, pause=False)
        if ok:
            task_meta["paused"] = False
            save_task_meta(task_meta)
            st.success(message)
        else:
            st.error(f"继续失败: {message}")
        st.rerun()

    if stop_clicked and task_meta:
        ok, message = kill_tracked_task(task_meta)
        if ok:
            task_meta["stop_requested"] = True
            task_meta["result_notified"] = True
            save_task_meta(task_meta)
            st.warning(f"已停止任务（PID={task_meta['pid']}）。")
        else:
            st.error(f"停止失败: {message}")
        st.rerun()

    with st.expander("运行状态与日志", expanded=True):
        if not task_meta:
            st.info("当前没有已登记任务。")
        else:
            log_path = Path(str(task_meta.get("log_path", "")))
            st.caption(f"日志文件：{log_path}")
            heartbeat_log_path_raw = str(task_meta.get("heartbeat_log_path", "")).strip()
            if heartbeat_log_path_raw:
                heartbeat_log_path = Path(heartbeat_log_path_raw)
            else:
                heartbeat_log_path = log_path.with_name(f"{log_path.stem}.heartbeat{log_path.suffix}")
            st.caption(f"心跳文件：{heartbeat_log_path}")
            if "cmd" in task_meta:
                st.caption(
                    "当前任务生效配置快照："
                    + " | ".join(
                        f"{k}={v}"
                        for k, v in extract_effective_config_from_cmd(task_meta["cmd"]).items()
                        if v
                    )
                )

            max_lines = int(st.session_state["ui_log_tail_lines"])
            logs = read_log_tail(log_path, max_lines=max(2000, max_lines * 4))
            heartbeat_logs = read_log_tail(heartbeat_log_path, max_lines=max(2000, max_lines * 4))
            keyword = str(st.session_state["ui_log_filter_keyword"]).strip().lower()
            if keyword:
                logs = [line for line in logs if keyword in line.lower()]
                heartbeat_logs = [line for line in heartbeat_logs if keyword in line.lower()]
            display_logs = logs[-max_lines:]
            display_heartbeat_logs = heartbeat_logs[-max_lines:]
            tab_normal, tab_heartbeat = st.tabs(["普通日志", "心跳日志"])
            with tab_normal:
                st.code("\n".join(display_logs) if display_logs else "(暂无普通日志)", language="text")
            with tab_heartbeat:
                st.code(
                    "\n".join(display_heartbeat_logs) if display_heartbeat_logs else "(暂无心跳日志)",
                    language="text",
                )

    if task_running and bool(st.session_state["ui_auto_refresh"]):
        time.sleep(int(st.session_state["ui_refresh_interval_sec"]))
        st.rerun()
