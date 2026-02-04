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
    assessAudio: async (audioBlob: Blob): Promise<AssessmentResult> => {
        const formData = new FormData();
        formData.append('audio', audioBlob, 'recording.wav');

        const response = await axios.post(`${API_BASE_URL}/assess-audio`, formData, {
            headers: {
                'Content-Type': 'multipart/form-data',
            },
        });
        return response.data;
    },

    // Health check
    healthCheck: async (): Promise<{ status: string }> => {
        const response = await axios.get(`${API_BASE_URL}/health`);
        return response.data;
    },
};
