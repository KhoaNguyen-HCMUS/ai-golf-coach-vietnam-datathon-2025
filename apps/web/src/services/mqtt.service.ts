import { toast } from 'sonner';

const BASE_URL = process.env.NEXT_PUBLIC_API_URL;
const MQTT_API_BASE_URL = `${BASE_URL}/api/mqtt`;

export interface MqttStartResponse {
  message: string;
}

export interface MqttStopResponse {
  message: string;
  videoUploaded?: boolean;
}

export interface MqttStopNoVideoResponse {
  message: string;
}

/**
 * Send START command to MQTT server
 * @returns Promise with response data
 */
export const sendStartCommand = async (): Promise<MqttStartResponse> => {
  try {
    const response = await fetch(`${MQTT_API_BASE_URL}/cmd/start`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
    });

    if (!response.ok) {
      const errorData = await response.json().catch(() => ({ error: 'Unknown error' }));
      throw new Error(errorData.error || `HTTP error! status: ${response.status}`);
    }

    const data = await response.json();
    console.log('MQTT START command sent successfully:', data);
    // Show success toast with message from response
    if (data.message) {
      toast.success(data.message);
    }
    return data;
  } catch (error: any) {
    console.error('Error sending MQTT START command:', error);
    // Show error toast
    toast.error(error.message || 'Failed to send START command');
    throw error;
  }
};

export const sendStopCommand = async (videoFile?: File): Promise<MqttStopResponse> => {
  try {
    const formData = new FormData();
    if (videoFile) {
      formData.append('video', videoFile);
    }

    const response = await fetch(`${MQTT_API_BASE_URL}/cmd/stop`, {
      method: 'POST',
      body: formData,
    });

    if (!response.ok) {
      const errorData = await response.json().catch(() => ({ error: 'Unknown error' }));
      throw new Error(errorData.error || `HTTP error! status: ${response.status}`);
    }

    const data = await response.json();
    console.log('MQTT STOP command sent successfully:', data);
    // Show success toast with message from response
    if (data.message) {
      toast.success(data.message);
    }
    return data;
  } catch (error: any) {
    console.error('Error sending MQTT STOP command:', error);
    // Show error toast
    toast.error(error.message || 'Failed to upload video');
    throw error;
  }
};

/**
 * Send STOP command without video to MQTT server
 * @returns Promise with response data
 */
export const sendStopNoVideoCommand = async (): Promise<MqttStopNoVideoResponse> => {
  try {
    const response = await fetch(`${MQTT_API_BASE_URL}/cmd/stop-no-video`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
    });

    if (!response.ok) {
      const errorData = await response.json().catch(() => ({ error: 'Unknown error' }));
      throw new Error(errorData.error || `HTTP error! status: ${response.status}`);
    }

    const data = await response.json();
    console.log('MQTT STOP (no video) command sent successfully:', data);
    // Show success toast with message from response
    if (data.message) {
      toast.success(data.message);
    }
    return data;
  } catch (error: any) {
    console.error('Error sending MQTT STOP (no video) command:', error);
    // Show error toast
    toast.error(error.message || 'Failed to send STOP command');
    throw error;
  }
};
