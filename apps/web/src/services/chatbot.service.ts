import { toast } from 'sonner';

const BASE_URL = process.env.NEXT_PUBLIC_API_URL;

export interface ChatbotRequest {
  history: string[];
  message: string;
}

export interface ChatbotResponse {
  message: string; // For toast notification
  data: string; // Chat response content
}

/**
 * Ask chatbot with conversation history
 * @param history Array of previous messages (max 10)
 * @param message Current user message
 * @returns Promise with chatbot response
 */
export const askChatbot = async (history: string[], message: string): Promise<ChatbotResponse> => {
  try {
    const response = await fetch(`${BASE_URL}/api/chatbot/ask`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        history: history.slice(-10), // Get last 10 messages
        message: message,
      }),
    });

    if (!response.ok) {
      const errorData = await response.json().catch(() => ({ error: 'Unknown error' }));
      throw new Error(errorData.error || `HTTP error! status: ${response.status}`);
    }

    const data = await response.json();
    console.log('Chatbot response received:', data);
    return data;
  } catch (error: any) {
    console.error('Error asking chatbot:', error);
    toast.error(error.message || 'Failed to get chatbot response');
    throw error;
  }
};
