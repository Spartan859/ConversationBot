"""
GPT-SoVITS TTS 使用示例
演示如何使用 src/tts/gpt_sovits_tts.py 进行语音合成
"""

from src.tts import GPTSoVITSTTS
import os


def example_1_basic_synthesis():
    """示例1：基础语音合成"""
    print("\n=== 示例1：基础语音合成 ===")
    
    # 初始化 TTS 客户端
    tts = GPTSoVITSTTS(
        api_url="http://127.0.0.1:8000",
        gpt_model_name="【GSVI】jianhua_tao-e10",
        sovits_model_name="【GSVI】jianhua_tao_e10_s540_l32",
        ref_audio_path="./custom_refs/jianhua_tao_1.wav_0000156800_0000330880.wav",
        ref_text="这个组委会的邀请啊，能有机会给大家做一些工作上的一些分享",
        ref_text_lang="中文"
    )
    
    # 合成单段文本
    text = "清华大学是一所综合性大学，位于北京市海淀区。"
    output_path = "outputs/example_1.wav"
    
    os.makedirs("outputs", exist_ok=True)
    audio_path = tts.synthesize(text, output_path)
    print(f"✓ 音频已保存: {audio_path}")


def example_2_multi_line_text():
    """示例2：多行文本合成（带停顿）"""
    print("\n=== 示例2：多行文本合成 ===")
    
    tts = GPTSoVITSTTS(
        api_url="http://127.0.0.1:8000",
        gpt_model_name="【GSVI】jianhua_tao-e10",
        sovits_model_name="【GSVI】jianhua_tao_e10_s540_l32",
        ref_audio_path="./custom_refs/jianhua_tao_1.wav_0000156800_0000330880.wav",
        ref_text="这个组委会的邀请啊，能有机会给大家做一些工作上的一些分享"
    )
    
    # 使用 \n 创建停顿
    text = """清华大学的校训是"自强不息，厚德载物"。

简单说，"自强不息"就是要永远努力向上、不服输。

"厚德载物"则是要有宽厚的品德，能包容、担当。"""
    
    output_path = "outputs/example_2.wav"
    audio_path = tts.synthesize(text, output_path)
    print(f"✓ 音频已保存: {audio_path}")


def example_3_parameter_tuning():
    """示例3：参数调优"""
    print("\n=== 示例3：参数调优 ===")
    
    tts = GPTSoVITSTTS(api_url="http://127.0.0.1:8000")
    
    text = "这是一个参数调优的示例。"
    
    # 场景1：快速语速
    print("  生成快速语速版本...")
    tts.synthesize(
        text=text,
        output_path="outputs/example_3_fast.wav",
        speed=1.5,  # 1.5倍速
        temperature=0.8
    )
    
    # 场景2：慢速、更稳定
    print("  生成慢速稳定版本...")
    tts.synthesize(
        text=text,
        output_path="outputs/example_3_slow.wav",
        speed=0.8,  # 0.8倍速
        temperature=0.6,  # 降低随机性
        top_k=5
    )
    
    print("✓ 两个版本都已生成")


def example_4_batch_synthesis():
    """示例4：批量合成"""
    print("\n=== 示例4：批量合成 ===")
    
    tts = GPTSoVITSTTS(
        api_url="http://127.0.0.1:8000",
        ref_audio_path="./custom_refs/jianhua_tao_1.wav_0000156800_0000330880.wav",
        ref_text="这个组委会的邀请啊，能有机会给大家做一些工作上的一些分享"
    )
    
    texts = [
        "清华大学位于北京市海淀区。",
        "清华有多个学院和研究所。",
        "清华培养了大批优秀人才。",
        "清华的校训是自强不息，厚德载物。"
    ]
    
    audio_files = tts.batch_synthesize(
        texts=texts,
        output_dir="outputs/batch",
        prefix="part"
    )
    
    successful = [f for f in audio_files if f]
    print(f"✓ 成功生成 {len(successful)}/{len(texts)} 个音频文件")


def example_5_get_models():
    """示例5：获取可用模型"""
    print("\n=== 示例5：获取可用模型 ===")
    
    tts = GPTSoVITSTTS(api_url="http://127.0.0.1:8000")
    
    try:
        models = tts.get_model_list()
        print(f"✓ 连接成功")
        print(f"  可用的 GPT 模型 ({len(models['gpt'])} 个):")
        for model in models['gpt']:
            print(f"    - {model}")
        print(f"  可用的 SoVITS 模型 ({len(models['sovits'])} 个):")
        for model in models['sovits']:
            print(f"    - {model}")
    except Exception as e:
        print(f"✗ 连接失败: {e}")
        print("  请确保 GPT-SoVITS-V4-Inference 服务已启动")


def example_6_cached_synthesis():
    """示例6：使用缓存加速"""
    print("\n=== 示例6：使用缓存加速 ===")
    
    import hashlib
    import time
    
    class CachedTTS:
        def __init__(self, tts, cache_dir="audio_cache"):
            self.tts = tts
            self.cache_dir = cache_dir
            os.makedirs(cache_dir, exist_ok=True)
        
        def synthesize(self, text: str) -> str:
            text_hash = hashlib.md5(text.encode()).hexdigest()
            cache_path = os.path.join(self.cache_dir, f"{text_hash}.wav")
            
            if os.path.exists(cache_path):
                print(f"  ✓ 使用缓存: {cache_path}")
                return cache_path
            
            print(f"  → 生成新音频...")
            self.tts.synthesize(text, output_path=cache_path)
            return cache_path
    
    # 创建缓存包装器
    tts = GPTSoVITSTTS(
        api_url="http://127.0.0.1:8000",
        ref_audio_path="./custom_refs/jianhua_tao_1.wav_0000156800_0000330880.wav",
        ref_text="这个组委会的邀请啊，能有机会给大家做一些工作上的一些分享"
    )
    cached_tts = CachedTTS(tts)
    
    text = "你好，有什么可以帮助你的吗？"
    
    # 首次调用
    print("第一次调用（生成新音频）:")
    start = time.time()
    audio1 = cached_tts.synthesize(text)
    time1 = time.time() - start
    print(f"  耗时: {time1:.2f} 秒")
    
    # 再次调用（使用缓存）
    print("\n第二次调用（使用缓存）:")
    start = time.time()
    audio2 = cached_tts.synthesize(text)
    time2 = time.time() - start
    print(f"  耗时: {time2:.2f} 秒")
    print(f"\n✓ 加速比: {time1/time2:.1f}x")


def main():
    """主函数：运行所有示例"""
    print("=" * 60)
    print("GPT-SoVITS TTS 使用示例")
    print("=" * 60)
    
    # 检查服务是否可用
    example_5_get_models()
    
    # 运行其他示例
    try:
        example_1_basic_synthesis()
        example_2_multi_line_text()
        example_3_parameter_tuning()
        example_4_batch_synthesis()
        example_6_cached_synthesis()
        
        print("\n" + "=" * 60)
        print("✓ 所有示例运行完成！")
        print("  生成的音频文件位于 outputs/ 目录")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n✗ 运行失败: {e}")
        print("\n请检查：")
        print("  1. GPT-SoVITS-V4-Inference 服务是否启动（python api.py）")
        print("  2. 模型文件是否正确配置")
        print("  3. 参考音频路径是否正确")


if __name__ == "__main__":
    main()
