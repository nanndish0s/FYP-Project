import type { InterviewQuestion } from '../types/interview';

export const INTERVIEW_QUESTIONS: InterviewQuestion[] = [
    {
        id: 'q1_curiosity',
        text: 'Describe a time when you explored a new technology or programming concept outside of your required coursework or job responsibilities. What motivated you to learn it?',
        targetTrait: 'curiosity',
        timeLimit: 60
    },
    {
        id: 'q2_critical_thinking',
        text: 'Walk me through your approach to debugging a complex system failure where multiple components might be involved. How do you systematically identify the root cause?',
        targetTrait: 'critical_thinking',
        timeLimit: 60
    },
    {
        id: 'q3_creativity',
        text: 'Tell me about a creative or unconventional solution you developed to solve a technical problem. What alternative approaches did you consider, and why did you choose your final solution?',
        targetTrait: 'creativity',
        timeLimit: 60
    }
];
