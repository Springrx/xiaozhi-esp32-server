<template>
  <div class="welcome">
    <HeaderBar/>
    <el-main style="padding: 16px;display: flex;flex-direction: column;">
      <div style="border-radius: 16px;background: #fafcfe; border: 1px solid #e8f0ff;">
        <div
            style="padding: 15px 24px;font-weight: 700;font-size: 19px;text-align: left;color: #3d4566;display: flex;gap: 13px;align-items: center;">
          <div
              style="width: 37px;height: 37px;background: #5778ff;border-radius: 50%;display: flex;align-items: center;justify-content: center;">
            <img loading="lazy" src="@/assets/home/setting-user.png" alt="" style="width: 19px;height: 19px;"/>
          </div>
          {{ form.agentName }}
        </div>
        <div style="height: 1px;background: #e8f0ff;"/>
        <el-form ref="form" :model="form" label-width="72px">
          <div style="padding: 16px 24px;max-width: 792px;">
            <el-form-item label="助手昵称：">
              <div class="input-46" style="width: 100%; max-width: 412px;">
                <el-input v-model="form.agentName"/>
              </div>
            </el-form-item>
            <el-form-item label="角色模版：">
              <div style="display: flex;gap: 8px;">
                <div v-for="template in templates" :key="template" class="template-item" :class="{ 'template-loading': loadingTemplate }" @click="selectTemplate(template)">
                  {{ template }}
                </div>
              </div>
            </el-form-item>
            <el-form-item label="角色音色：">
              <div style="display: flex;gap: 8px;align-items: center;">
                <div class="input-46" style="flex:1.4;">
                  <el-select v-model="form.ttsVoiceId" placeholder="请选择" style="width: 100%;">
                    <el-option v-for="item in options" :key="item.value" :label="item.label"
                               :value="item.value">
                    </el-option>
                  </el-select>
                </div>
                <div class="audio-box">
                  <audio src="http://music.163.com/song/media/outer/url?id=447925558.mp3" controls
                         style="height: 100%;width: 100%;"/>
                </div>
              </div>
            </el-form-item>
            <el-form-item label="角色介绍：">
              <div class="textarea-box">
                <el-input type="textarea" rows="5" resize="none" placeholder="请输入内容"
                          v-model="form.systemPrompt" maxlength="2000" show-word-limit/>
              </div>
            </el-form-item>
            <el-form-item label="记忆体：">
              <div class="textarea-box">
                <el-input type="textarea" rows="5" resize="none" placeholder="请输入内容"
                          v-model="form.langCode" maxlength="1000"/>
                <div class="prompt-bottom" @click="clearMemory">
                  <div style="display: flex;gap: 8px;align-items: center;">
                    <div style="color: #979db1;font-size: 11px;">当前记忆（每次对话后重新生成）</div>
                    <div class="clear-btn">
                      <i class="el-icon-delete-solid" style="font-size: 11px;"/>
                      清除
                    </div>
                  </div>
                  <div style="color: #979db1;font-size:11px;">{{ form.langCode.length }}/1000</div>
                </div>
              </div>
            </el-form-item>
            <el-form-item v-for="model in models" :key="model.label" :label="model.label" class="model-item">
              <el-select v-model="form.model[model.key]" filterable placeholder="请选择" class="select-field">
                <el-option v-for="item in options" :key="item.value" :label="item.label" :value="item.value"/>
              </el-select>
            </el-form-item>
            <el-form-item label="" class="lh-form-item" style="margin-top: -25px;">
              <div style="color: #979db1;text-align: left;">除了“Qwen
                实时”，其他模型通常会增加约1秒的延迟。改变模型后，建议清空记忆体，以免影响体验。
              </div>
            </el-form-item>
          </div>
        </el-form>
        <div style="display: flex;padding: 16px;gap: 8px;align-items: center;">
          <div class="save-btn" @click="saveConfig">
            保存配置
          </div>
          <div class="reset-btn" @click="resetConfig">
            重制
          </div>
          <div class="clear-text">
            <img loading="lazy" src="@/assets/home/red-info.png" alt="" style="width: 19px;height: 19px;"/>
            保存配置后，需要重启设备，新的配置才会生效。
          </div>
        </div>
      </div>
      <div style="font-size: 12px;font-weight: 400;margin-top: auto;padding-top: 24px;color: #979db1;">
        ©2025 xiaozhi-esp32-server
      </div>
    </el-main>
  </div>
</template>

<script>
import HeaderBar from "@/components/HeaderBar.vue";

export default {
  name: 'RoleConfigPage',
  components: {HeaderBar},
  data() {
    return {
      form: {
        agentCode: "",
        agentName: "",
        ttsVoiceId: "",
        systemPrompt: "",
        langCode: "",
        language: "",
        sort: "",
        model: {
          ttsModelId: "",
          vadModelId: "",
          asrModelId: "",
          llmModelId: "",
          memModelId: "",
          intentModelId: "",
        }
      },
      options: [
        {value: '选项1', label: '黄金糕'},
        {value: '选项2', label: '双皮奶'}
      ],
      models: [
        {label: '大语言模型(LLM)', key: 'llmModelId'},
        {label: '语音识别(ASR)', key: 'asrModelId'},
        {label: '语音活动检测模型(VAD)', key: 'vadModelId'},
        {label: '语音合成模型(TTS)', key: 'ttsModelId'},
        {label: '意图识别模型(Intent)', key: 'intentModelId'},
        {label: '记忆模型(Memory)', key: 'memModelId'}
      ],
      templates: ['台湾女友', '土豆子', '英语老师', '好奇小男孩', '汪汪队队长'],
      loadingTemplate: false
    }
  },
  methods: {
    saveConfig() {
      const configData = {
        agentCode: this.form.agentCode,
        agentName: this.form.agentName,
        asrModelId: this.form.model.asrModelId,
        vadModelId: this.form.model.vadModelId,
        llmModelId: this.form.model.llmModelId,
        ttsModelId: this.form.model.ttsModelId,
        ttsVoiceId: this.form.ttsVoiceId,
        memModelId: this.form.model.memModelId,
        intentModelId: this.form.model.intentModelId,
        systemPrompt: this.form.systemPrompt,
        langCode: this.form.langCode,
        language: this.form.language,
        sort: this.form.sort
      };
      import('@/apis/module/user').then(({default: userApi}) => {
        userApi.updateAgentConfig(this.$route.query.agentId, configData, ({data}) => {
          if (data.code === 0) {
            this.$message.success('配置保存成功');
          } else {
            this.$message.error(data.msg || '配置保存失败');
          }
        });
      });
    },
    resetConfig() {
      this.$confirm('确定要重置配置吗？', '提示', {
        confirmButtonText: '确定',
        cancelButtonText: '取消',
        type: 'warning'
      }).then(() => {
        // 重置表单
        this.form = {
          agentCode: "",
          agentName: "",
          ttsVoiceId: "",
          systemPrompt: "",
          langCode: "",
          language: "",
          sort: "",
          model: {
            ttsModelId: "",
            vadModelId: "",
            asrModelId: "",
            llmModelId: "",
            memModelId: "",
            intentModelId: "",
          }
        }
        this.$message.success('配置已重置')
      }).catch(() => {
      })
    },
    selectTemplate(templateName) {
      if (this.loadingTemplate) return;

      this.loadingTemplate = true;
      import('@/apis/module/user').then(({default: userApi}) => {
        userApi.getAgentTemplate(
            {templateName},
            (response) => {
              this.loadingTemplate = false;
              if (response.data.code === 0 && response.data.data.length > 0) {
                this.applyTemplateData(response.data.data[0]);
                this.$message.success(`「${templateName}」模板已应用`);
              } else {
                this.$message.warning(`未找到「${templateName}」模板`);
              }
            }
        );
      }).catch((error) => {
        this.loadingTemplate = false;
        this.$message.error('模板加载失败');
        console.error('接口异常:', error);
      });
    },
    applyTemplateData(templateData) {
      this.form = {
        ...this.form,
        agentName: templateData.agentName || this.form.agentName,
        ttsVoiceId: templateData.ttsVoiceId || this.form.ttsVoiceId,
        systemPrompt: templateData.systemPrompt || this.form.systemPrompt,
        langCode: templateData.langCode || this.form.langCode,
        model: {
          ttsModelId: templateData.ttsModelId || this.form.model.ttsModelId,
          vadModelId: templateData.vadModelId || this.form.model.vadModelId,
          asrModelId: templateData.asrModelId || this.form.model.asrModelId,
          llmModelId: templateData.llmModelId || this.form.model.llmModelId,
          memModelId: templateData.memModelId || this.form.model.memModelId,
          intentModelId: templateData.intentModelId || this.form.model.intentModelId
        }
      };
    },
  fetchAgentConfig(agentId) {
    import('@/apis/module/user').then(({default: userApi}) => {
      userApi.getDeviceConfig(agentId, ({data}) => {
        if (data.code === 0) {
          this.form = {
            ...this.form,
            ...data.data,
            model: {
              ttsModelId: data.data.ttsModelId,
              vadModelId: data.data.vadModelId,
              asrModelId: data.data.asrModelId,
              llmModelId: data.data.llmModelId,
              memModelId: data.data.memModelId,
              intentModelId: data.data.intentModelId
            }
          };
        } else {
          this.$message.error(data.msg || '获取配置失败');
        }
      });
    });
  },
  // 清空记忆体内容
  clearMemory() {
    this.form.langCode = "";
    this.$message.success("记忆体已清空");
  },
},
  mounted() {
    const agentId = this.$route.query.agentId;
    console.log('agentId2222',agentId);
    if (agentId) {
      this.fetchAgentConfig(agentId);
    }
  }
}
</script>

<style scoped>
.welcome {
  min-width: 900px;
  min-height: 506px;
  height: 100vh;
  display: flex;
  flex-direction: column;
  background-image: url("@/assets/home/background.png");
  background-size: cover;
  /* 确保背景图像覆盖整个元素 */
  background-position: center;
  /* 从顶部中心对齐 */
  -webkit-background-size: cover;
  /* 兼容老版本WebKit浏览器 */
  -o-background-size: cover;
  /* 兼容老版本Opera浏览器 */
}

.el-form-item ::v-deep .el-form-item__label {
  font-size: 10px !important;
  color: #3d4566 !important;
  font-weight: 400;
  line-height: 22px;
  padding-bottom: 2px;
}

.select-field{
  width: 100%;
  max-width: 720px;
  border: 1px solid #e4e6ef;
  background: #f6f8fb;
  border-radius: 8px;
  height: 36px !important;
}

.audio-box {
  flex: 1;
  height: 37px;
  border-radius: 20px;
  border: 1px solid #e4e6ef;
}

.clear-btn {
  width: 48px;
  height: 19px;
  background: #fd8383;
  border-radius: 10px;
  line-height: 19px;
  font-size: 11px;
  color: #fff;
  cursor: pointer;
}

.clear-text {
  color: #979db1;
  font-size: 11px;
  display: flex;
  align-items: center;
  gap: 8px;
  margin-left: 16px;
}

.template-item {
  height: 37px;
  width: 76px;
  border-radius: 8px;
  background: #e6ebff;
  line-height: 37px;
  font-weight: 400;
  font-size: 11px;
  text-align: center;
  color: #5778ff;
  cursor: pointer;
  transition: background-color 0.3s ease;
}

.template-item:hover {
  background-color: #d0d8ff;
}

.prompt-bottom {
  margin-bottom: 4px;
  display: flex;
  justify-content: space-between;
  padding: 0 16px;
  align-items: center;
}

.input-46 {
  border: 1px solid #e4e6ef;
  background: #f6f8fb;
  border-radius: 8px;
  height: 36px !important;
}

.save-btn,
.reset-btn {
  width: 112px;
  height: 37px;
  border-radius: 18px;
  line-height: 37px;
  box-sizing: border-box;
  cursor: pointer;
  font-size: 11px
}

.save-btn {
  border-radius: 18px;
  background: #5778ff;
  color: #fff;
}

.reset-btn {
  border: 1px solid #adbdff;
  background: #e6ebff;
  color: #5778ff;
}

.textarea-box {
  border: 1px solid #e4e6ef;
  border-radius: 8px;
  background: #f6f8fb;
}
</style>

