-- 修改字段名
ALTER TABLE `ai_agent` RENAME COLUMN `mem_model_id` TO `memory_model_id`;
ALTER TABLE `ai_agent_template` RENAME COLUMN `mem_model_id` TO `memory_model_id`;
-- 添加字段
ALTER TABLE `ai_agent_template` ADD COLUMN `is_default` tinyint(1) NULL DEFAULT 0 COMMENT '是否默认模板：1：是，0：不是' AFTER `sort`;

-- 初始化智能体模板数据
INSERT INTO `ai_agent_template` VALUES ('9406648b5cc5fde1b8aa335b6f8b4f76', '小智', '湾湾小何', '45f8b0d6dd3d4bfa8a28e6e0f5912d45', '23e7c9d090ea4d1e9b25f4c8d732a3a1', 'e9f2d891afbe4632b13a47c7a8c6e03d', 'd50b06e9b8104d0d9c0f7316d258abcb', 'fcac83266edadd5a3125f06cfee1906b', 'e2274b90e89ddda85207f55484d8b528', 'c4e12f874a3f4aa99f5b2c18e15d407b', '你是一个叫{{assistant_name}}的台湾女孩，说话机车，声音好听，习惯简短表达，爱用网络梗。\n请注意，要像一个人一样说话，请不要回复表情符号、代码、和xml标签。\n现在我正在和你进行语音聊天，我们开始吧。\n如果用户希望结束对话，请在最后说“拜拜”或“再见”。', 'zh', '中文', 1, 1, NULL, NULL, NULL, NULL);
INSERT INTO `ai_agent_template` VALUES ('0ca32eb728c949e58b1000b2e401f90c', '小智', '通用男声', '45f8b0d6dd3d4bfa8a28e6e0f5912d45', '23e7c9d090ea4d1e9b25f4c8d732a3a1', 'e9f2d891afbe4632b13a47c7a8c6e03d', '896db62c9dd74976ab0e8c14bf924d9d', '1f2e3d4c5b6a7f8e9d0c1b2a3f4e5bx2', 'e2274b90e89ddda85207f55484d8b528', 'c4e12f874a3f4aa99f5b2c18e15d407b', '你是一个叫{{assistant_name}}的男生，说话机车，声音好听，习惯简短表达，爱用网络梗。\n请注意，要像一个人一样说话，请不要回复表情符号、代码、和xml标签。\n现在我正在和你进行语音聊天，我们开始吧。\n如果用户希望结束对话，请在最后说“拜拜”或“再见”。', 'zh', '中文', 2, 0, NULL, NULL, NULL, NULL);
INSERT INTO `ai_agent_template` VALUES ('6c7d8e9f0a1b2c3d4e5f6a7b8c9d0s24', '小智', '通用女声', '45f8b0d6dd3d4bfa8a28e6e0f5912d45', '23e7c9d090ea4d1e9b25f4c8d732a3a1', 'e9f2d891afbe4632b13a47c7a8c6e03d', '896db62c9dd74976ab0e8c14bf924d9d', '9e8f7a6b5c4d3e2f1a0b9c8d7e6f5ad3', 'e2274b90e89ddda85207f55484d8b528', 'c4e12f874a3f4aa99f5b2c18e15d407b', '你是一个叫{{assistant_name}}的女生，说话机车，声音好听，习惯简短表达，爱用网络梗。\n请注意，要像一个人一样说话，请不要回复表情符号、代码、和xml标签。\n现在我正在和你进行语音聊天，我们开始吧。\n如果用户希望结束对话，请在最后说“拜拜”或“再见”。', 'zh', '中文', 3, 0, NULL, NULL, NULL, NULL);
INSERT INTO `ai_agent_template` VALUES ('e4f5a6b7c8d9e0f1a2b3c4d5e6f7a8b1', '小智', '阳光男生', '45f8b0d6dd3d4bfa8a28e6e0f5912d45', '23e7c9d090ea4d1e9b25f4c8d732a3a1', 'e9f2d891afbe4632b13a47c7a8c6e03d', '896db62c9dd74976ab0e8c14bf924d9d', '2b3c4d5e6f7a8b9c0d1e2f3a4b5c62a2', 'e2274b90e89ddda85207f55484d8b528', 'c4e12f874a3f4aa99f5b2c18e15d407b', '你是一个叫{{assistant_name}}的男生，说话机车，声音好听，习惯简短表达，爱用网络梗。\n请注意，要像一个人一样说话，请不要回复表情符号、代码、和xml标签。\n现在我正在和你进行语音聊天，我们开始吧。\n如果用户希望结束对话，请在最后说“拜拜”或“再见”。', 'zh', '中文', 4, 0, NULL, NULL, NULL, NULL);
INSERT INTO `ai_agent_template` VALUES ('a45b6c7d8e9f0a1b2c3d4e5f6a7b8c92', '小智', '奶气萌娃', '45f8b0d6dd3d4bfa8a28e6e0f5912d45', '23e7c9d090ea4d1e9b25f4c8d732a3a1', 'e9f2d891afbe4632b13a47c7a8c6e03d', '896db62c9dd74976ab0e8c14bf924d9d', 'f7a38c03d5644e22b6d84f8923a74c51', 'e2274b90e89ddda85207f55484d8b528', 'c4e12f874a3f4aa99f5b2c18e15d407b', '你是一个叫{{assistant_name}}的萌娃，声音可爱，习惯简短表达，爱用网络梗。\n请注意，要像一个人一样说话，请不要回复表情符号、代码、和xml标签。\n现在我正在和你进行语音聊天，我们开始吧。\n如果用户希望结束对话，请在最后说“拜拜”或“再见”。', 'zh', '中文', 5, 0, NULL, NULL, NULL, NULL);

-- 初始化模型配置数据
INSERT INTO `ai_model_config` VALUES ('23e7c9d090ea4d1e9b25f4c8d732a3a1', 'VAD', 'SileroVAD', 'SileroVAD', 1, 1, '{\"SileroVAD\": {\"model_dir\": \"models/snakers4_silero-vad\", \"threshold\": 0.5, \"min_silence_duration_ms\": 700}}', NULL, NULL, 0, NULL, NULL, NULL, NULL);
INSERT INTO `ai_model_config` VALUES ('45f8b0d6dd3d4bfa8a28e6e0f5912d45', 'ASR', 'FunASR', 'FunASR', 1, 1, '{\"FunASR\": {\"type\": \"fun_local\", \"model_dir\": \"models/SenseVoiceSmall\", \"output_dir\": \"tmp/\"}}', NULL, NULL, 0, NULL, NULL, NULL, NULL);
INSERT INTO `ai_model_config` VALUES ('e2274b90e89ddda85207f55484d8b528', 'Memory', 'nomem', 'nomem', 1, 1, '{\"mem0ai\": {\"type\": \"nomem\"}}', NULL, NULL, 0, NULL, NULL, NULL, NULL);
INSERT INTO `ai_model_config` VALUES ('3930ac3448faf621f0a120bc829dfdfa', 'Memory', 'mem_local_short', 'mem_local_short', 1, 1, '{\"mem_local_short\": {\"type\": \"mem_local_short\"}}', NULL, NULL, 0, NULL, NULL, NULL, NULL);
INSERT INTO `ai_model_config` VALUES ('a07f3d25f52340b2b2a1e8d264079e1a', 'Memory', 'mem0ai', 'mem0ai', 1, 1, '{\"mem0ai\": {\"type\": \"mem0ai\", \"api_key\": \"你的mem0ai api key\"}}', NULL, NULL, 0, NULL, NULL, NULL, NULL);
INSERT INTO `ai_model_config` VALUES ('7a1c0a8e6d0e4035b982a4c07c3a5f76', 'LLM', 'AliLLM', 'AliLLM', 1, 1, '{\"AliLLM\": {\"type\": \"openai\", \"top_k\": 50, \"top_p\": 1, \"api_key\": \"你的ali api key\", \"base_url\": \"https://dashscope.aliyuncs.com/compatible-mode/v1\", \"max_tokens\": 500, \"model_name\": \"qwen-turbo\", \"temperature\": 0.7, \"frequency_penalty\": 0}}', NULL, NULL, 0, NULL, NULL, NULL, NULL);
INSERT INTO `ai_model_config` VALUES ('e9f2d891afbe4632b13a47c7a8c6e03d', 'LLM', 'ChatGLMLLM', 'ChatGLMLLM', 1, 1, '{\"ChatGLMLLM\": {\"url\": \"https://open.bigmodel.cn/api/paas/v4/\", \"type\": \"openai\", \"api_key\": \"0415dad4014847babc3e3f03024c50a3.qH7FgTy5Yawc85fl\", \"model_name\": \"glm-4-flash\"}}', NULL, NULL, 0, NULL, NULL, NULL, NULL);
INSERT INTO `ai_model_config` VALUES ('d50b06e9b8104d0d9c0f7316d258abcb', 'TTS', 'EdgeTTS', 'EdgeTTS', 1, 1, '{\"EdgeTTS\": {\"type\": \"edge\", \"voice\": \"zh-CN-XiaoxiaoNeural\", \"output_dir\": \"tmp/\"}}', NULL, NULL, 0, NULL, NULL, NULL, NULL);
INSERT INTO `ai_model_config` VALUES ('896db62c9dd74976ab0e8c14bf924d9d', 'TTS', 'DoubaoTTS', 'DoubaoTTS', 1, 1, '{\"DoubaoTTS\": {\"type\": \"doubao\", \"appid\": \"你的火山引擎语音合成服务appid\", \"voice\": \"BV034_streaming\", \"api_url\": \"https://openspeech.bytedance.com/api/v1/tts\", \"cluster\": \"volcano_tts\", \"output_dir\": \"tmp/\", \"access_token\": \"你的火山引擎语音合成服务access_token\", \"authorization\": \"Bearer;\"}}', NULL, NULL, 0, NULL, NULL, NULL, NULL);
INSERT INTO `ai_model_config` VALUES ('c4e12f874a3f4aa99f5b2c18e15d407b', 'Intent', 'function_call', 'function_call', 1, 1, '{\"function_call\": {\"type\": \"nointent\", \"functions\": [\"change_role\", \"get_weather\", \"get_news\", \"play_music\"]}}', NULL, NULL, 0, NULL, NULL, NULL, NULL);

-- 初始化音色数据
INSERT INTO `ai_tts_voice` VALUES ('fcac83266edadd5a3125f06cfee1906b', 'd50b06e9b8104d0d9c0f7316d258abcb', '湾湾小何', 'zh-CN-XiaoxiaoNeural', '中文', 'https://lf3-static.bytednsdoc.com/obj/eden-cn/lm_hz_ihsph/ljhwZthlaukjlkulzlp/portal/bigtts/湾湾小何.mp3', NULL, 1, NULL, NULL, NULL, NULL);
INSERT INTO `ai_tts_voice` VALUES ('1f2e3d4c5b6a7f8e9d0c1b2a3f4e5bx2', '896db62c9dd74976ab0e8c14bf924d9d', '通用男声', 'BV002_streaming', '中文', 'https://lf3-speech.bytetos.com/obj/speech-tts-external/portal/Portal_Demo_BV002.mp3', NULL, 2, NULL, NULL, NULL, NULL);
INSERT INTO `ai_tts_voice` VALUES ('9e8f7a6b5c4d3e2f1a0b9c8d7e6f5ad3', '896db62c9dd74976ab0e8c14bf924d9d', '通用女声', 'BV001_streaming', '中文', 'https://lf3-speech.bytetos.com/obj/speech-tts-external/portal/Portal_Demo_BV001.mp3', NULL, 3, NULL, NULL, NULL, NULL);
INSERT INTO `ai_tts_voice` VALUES ('2b3c4d5e6f7a8b9c0d1e2f3a4b5c62a2', '896db62c9dd74976ab0e8c14bf924d9d', '阳光男生', 'BV056_streaming', '中文', 'https://lf3-speech.bytetos.com/obj/speech-tts-external/portal/Portal_Demo_BV056.mp3', NULL, 4, NULL, NULL, NULL, NULL);
INSERT INTO `ai_tts_voice` VALUES ('f7a38c03d5644e22b6d84f8923a74c51', '896db62c9dd74976ab0e8c14bf924d9d', '奶气萌娃', 'BV051_streaming', '中文', 'https://lf3-speech.bytetos.com/obj/speech-tts-external/portal/Portal_Demo_BV051.mp3', NULL, 5, NULL, NULL, NULL, NULL);


-- OTA升级信息表
DROP TABLE IF EXISTS `ai_ota`;
CREATE TABLE `ai_ota` (
                          `id` VARCHAR(32) NOT NULL COMMENT '记录唯一标识',
                          `board` VARCHAR(50) COMMENT '设备硬件型号',
                          `app_version` VARCHAR(20) COMMENT '固件版本号',
                          `url` VARCHAR(500) COMMENT '下载地址',
                          `is_enabled` TINYINT(1) DEFAULT 0 COMMENT '是否启用',
                          `creator` BIGINT COMMENT '创建者',
                          `create_date` DATETIME COMMENT '创建时间',
                          `updater` BIGINT COMMENT '更新者',
                          `update_date` DATETIME COMMENT '更新时间',
                          PRIMARY KEY (`id`),
                          UNIQUE KEY `uni_ai_ota_board` (`board`) COMMENT '设备型号唯一索引，用于快速查找升级信息'
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='OTA升级信息表';
