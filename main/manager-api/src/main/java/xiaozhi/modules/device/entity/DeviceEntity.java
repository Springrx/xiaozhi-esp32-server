package xiaozhi.modules.device.entity;

import java.util.Date;

import com.baomidou.mybatisplus.annotation.TableName;

import io.swagger.v3.oas.annotations.media.Schema;
import lombok.Data;
import lombok.EqualsAndHashCode;
import xiaozhi.common.entity.BaseEntity;

@Data
@EqualsAndHashCode(callSuper = false)
@TableName("ai_device")
@Schema(description = "设备信息")
public class DeviceEntity extends BaseEntity {
    @Schema(description = "关联用户ID")
    private Long userId;

    @Schema(description = "MAC地址")
    private String macAddress;

    @Schema(description = "最后连接时间")
    private Date lastConnectedAt;

    @Schema(description = "自动更新开关(0关闭/1开启)")
    private Integer autoUpdate;

    @Schema(description = "设备硬件型号")
    private String board;

    @Schema(description = "设备别名")
    private String alias;

    @Schema(description = "智能体ID")
    private String agentId;

    @Schema(description = "固件版本号")
    private String appVersion;

    @Schema(description = "排序")
    private Integer sort;

    @Schema(description = "更新者")
    private Long updater;

    @Schema(description = "更新时间")
    private Date updateDate;
}