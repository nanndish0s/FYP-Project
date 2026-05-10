export interface InterviewQuestion {
    id: string;
    text: string;
    targetTrait: 'curiosity' | 'critical_thinking' | 'creativity';
    timeLimit: number; // seconds
}

export interface ShapEntry {
    feature: string;
    impact: number;
    direction: 'positive' | 'negative';
}

export interface QuestionResponse {
    questionId: string;
    questionText: string;
    transcript: string;
    wordCount: number;
    scores: {
        curiosity: number;
        critical_thinking: number;
        creativity: number;
    };
    average: number;
    explanations?: {
        curiosity: ShapEntry[];
        critical_thinking: ShapEntry[];
        creativity: ShapEntry[];
    };
}

export interface InterviewSession {
    currentQuestionIndex: number;
    responses: QuestionResponse[];
    isComplete: boolean;
}

export interface FinalResults {
    responses: QuestionResponse[];
    aggregateScores: {
        curiosity: number;
        critical_thinking: number;
        creativity: number;
    };
    overallAverage: number;
    recommendation: string;
}
