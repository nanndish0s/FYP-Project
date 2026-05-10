import axios from 'axios';

const API_BASE_URL = 'http://localhost:5000/api';

export type Candidate = {
    id: string;
    transcript: string;
    word_count: number;
    scores: {
        curiosity: number;
        critical_thinking: number;
        creativity: number;
    };
}

export type ShapEntry = {
    feature: string;
    impact: number;
    direction: 'positive' | 'negative';
}

export type AssessmentResult = {
    transcript: string;
    word_count: number;
    scores: {
        curiosity: number;
        critical_thinking: number;
        creativity: number;
    };
    average: number;
    recommendation: string;
    explanations?: {
        curiosity: ShapEntry[];
        critical_thinking: ShapEntry[];
        creativity: ShapEntry[];
    };
}

export const api = {
    // Get all candidates
    getCandidates: async (): Promise<Candidate[]> => {
        const response = await axios.get(`${API_BASE_URL}/candidates`);
        return response.data;
    },

    // Get single candidate
    getCandidate: async (id: string): Promise<Candidate> => {
        const response = await axios.get(`${API_BASE_URL}/candidate/${id}`);
        return response.data;
    },

    // Assess audio file
    assessAudio: async (audioBlob: Blob, questionId?: string): Promise<AssessmentResult> => {
        const formData = new FormData();
        formData.append('audio', audioBlob, 'recording.wav');
        if (questionId) formData.append('question_id', questionId);

        const response = await axios.post(`${API_BASE_URL}/assess-audio`, formData, {
            headers: {
                'Content-Type': 'multipart/form-data',
            },
        });
        return response.data;
    },

    // Assess text input
    assessText: async (text: string, questionId?: string): Promise<AssessmentResult> => {
        const response = await axios.post(`${API_BASE_URL}/assess-text`, { text, question_id: questionId });
        return response.data;
    },

    // Health check
    healthCheck: async (): Promise<{ status: string }> => {
        const response = await axios.get(`${API_BASE_URL}/health`);
        return response.data;
    },
};
