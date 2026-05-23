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

    // Session storage
    saveSesssion: async (payload: object): Promise<{ id: string; created_at: string }> => {
        const response = await axios.post(`${API_BASE_URL}/sessions`, payload);
        return response.data;
    },

    listSessions: async (): Promise<SessionSummary[]> => {
        const response = await axios.get(`${API_BASE_URL}/sessions`);
        return response.data;
    },

    getSession: async (id: string): Promise<SessionDetail> => {
        const response = await axios.get(`${API_BASE_URL}/sessions/${id}`);
        return response.data;
    },

    deleteSession: async (id: string): Promise<void> => {
        await axios.delete(`${API_BASE_URL}/sessions/${id}`);
    },
};

export type SessionSummary = {
    id: string;
    created_at: string;
    recommendation: string;
    overall_average: number;
    curiosity_avg: number;
    critical_thinking_avg: number;
    creativity_avg: number;
};

export type SessionDetail = SessionSummary & {
    responses: {
        id: number;
        question_id: string;
        question_text: string;
        transcript: string;
        word_count: number;
        curiosity_score: number;
        critical_thinking_score: number;
        creativity_score: number;
        average_score: number;
        explanations: Record<string, ShapEntry[]>;
    }[];
};
