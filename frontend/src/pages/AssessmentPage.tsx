import React, { useState } from 'react';
import { AudioRecorder } from '../components/AudioRecorder';
import { GaugeChart } from '../components/GaugeChart';
import { api } from '../services/api';
import type { InterviewSession, QuestionResponse, FinalResults } from '../types/interview';
import { INTERVIEW_QUESTIONS } from '../data/interviewQuestions';
import { CheckCircle, AlertCircle, ChevronRight, Play, Mic, Sparkles, ArrowRight } from 'lucide-react';

type WizardStep = 'welcome' | 'question' | 'processing' | 'results';

export const AssessmentPage: React.FC = () => {
    const [step, setStep] = useState<WizardStep>('welcome');
    const [session, setSession] = useState<InterviewSession>({
        currentQuestionIndex: 0,
        responses: [],
        isComplete: false
    });
    const [currentRecording, setCurrentRecording] = useState<Blob | null>(null);
    const [isProcessing, setIsProcessing] = useState(false);
    const [finalResults, setFinalResults] = useState<FinalResults | null>(null);
    const [error, setError] = useState<string | null>(null);

    const currentQuestion = INTERVIEW_QUESTIONS[session.currentQuestionIndex];
    const isLastQuestion = session.currentQuestionIndex === INTERVIEW_QUESTIONS.length - 1;

    const handleStart = () => {
        setStep('question');
        setSession({
            currentQuestionIndex: 0,
            responses: [],
            isComplete: false
        });
    };

    const handleRecordingComplete = (audioBlob: Blob) => {
        setCurrentRecording(audioBlob);
    };

    const handleReRecord = () => {
        setCurrentRecording(null);
        setError(null);
    };

    const handleNext = async () => {
        if (!currentRecording) return;

        setIsProcessing(true);
        setError(null);

        try {
            const result = await api.assessAudio(currentRecording);

            const response: QuestionResponse = {
                questionId: currentQuestion.id,
                questionText: currentQuestion.text,
                transcript: result.transcript,
                wordCount: result.word_count,
                scores: result.scores,
                average: result.average
            };

            const newResponses = [...session.responses, response];

            if (isLastQuestion) {
                setStep('processing');

                const aggregateScores = {
                    curiosity: newResponses.reduce((sum, r) => sum + r.scores.curiosity, 0) / newResponses.length,
                    critical_thinking: newResponses.reduce((sum, r) => sum + r.scores.critical_thinking, 0) / newResponses.length,
                    creativity: newResponses.reduce((sum, r) => sum + r.scores.creativity, 0) / newResponses.length,
                };

                const overallAverage = (aggregateScores.curiosity + aggregateScores.critical_thinking + aggregateScores.creativity) / 3;

                let recommendation = 'NOT_RECOMMENDED';
                if (overallAverage >= 4.0) recommendation = 'STRONG_HIRE';
                else if (overallAverage >= 3.5) recommendation = 'RECOMMENDED';
                else if (overallAverage >= 3.0) recommendation = 'CONSIDER';

                setFinalResults({
                    responses: newResponses,
                    aggregateScores,
                    overallAverage,
                    recommendation
                });

                setTimeout(() => {
                    setStep('results');
                    setIsProcessing(false);
                }, 1500);
            } else {
                setSession({
                    currentQuestionIndex: session.currentQuestionIndex + 1,
                    responses: newResponses,
                    isComplete: false
                });
                setCurrentRecording(null);
                setIsProcessing(false);
            }
        } catch (err) {
            setError('Failed to process audio. Please try again.');
            console.error(err);
            setIsProcessing(false);
        }
    };

    const handleReset = () => {
        setStep('welcome');
        setSession({
            currentQuestionIndex: 0,
            responses: [],
            isComplete: false
        });
        setCurrentRecording(null);
        setFinalResults(null);
        setError(null);
    };

    return (
        <div className="min-h-screen bg-gradient-to-br from-gray-50 to-blue-50 py-12">
            <div className="container mx-auto px-4 max-w-5xl">
                {step === 'welcome' && (
                    <div className="text-center space-y-8 animate-fadeIn">
                        <div>
                            <div className="inline-flex items-center gap-2 px-4 py-2 bg-primary-100 text-primary-700 rounded-full mb-6 shadow-sm">
                                <Mic className="w-4 h-4" />
                                <span className="text-sm font-semibold">Structured Interview Assessment</span>
                            </div>

                            <h1 className="text-5xl md:text-6xl font-extrabold mb-6 text-gray-900">
                                Live C3 Skills
                                <span className="block bg-gradient-to-r from-primary-600 to-violet-600 bg-clip-text text-transparent">
                                    Assessment
                                </span>
                            </h1>

                            <p className="text-xl md:text-2xl text-gray-600 mb-4">
                                Answer <strong className="text-primary-600">{INTERVIEW_QUESTIONS.length} questions</strong> to evaluate your soft skills
                            </p>
                            <p className="text-gray-500">
                                Powered by Advanced AI • Fully Objective Results
                            </p>
                        </div>

                        <div className="card max-w-3xl mx-auto text-left shadow-xl border-2 border-gray-100 hover:shadow-2xl transition-shadow">
                            <div className="flex items-center gap-3 mb-6 pb-6 border-b border-gray-100">
                                <div className="w-12 h-12 bg-gradient-to-br from-primary-500 to-violet-500 rounded-xl flex items-center justify-center">
                                    <Play className="w-6 h-6 text-white" />
                                </div>
                                <div>
                                    <h2 className="text-2xl font-bold text-gray-900">How It Works</h2>
                                    <p className="text-sm text-gray-500">Quick overview before you begin</p>
                                </div>
                            </div>

                            <div className="grid md:grid-cols-3 gap-6 mb-8">
                                <div className="text-center p-4 bg-blue-50 rounded-xl">
                                    <div className="w-12 h-12 bg-blue-500 text-white rounded-full flex items-center justify-center mx-auto mb-3 font-bold text-lg">
                                        {INTERVIEW_QUESTIONS.length}
                                    </div>
                                    <div className="font-semibold text-gray-900">Questions</div>
                                    <div className="text-sm text-gray-600">Targeting C3 skills</div>
                                </div>
                                <div className="text-center p-4 bg-purple-50 rounded-xl">
                                    <div className="w-12 h-12 bg-purple-500 text-white rounded-full flex items-center justify-center mx-auto mb-3 font-bold text-lg">
                                        60s
                                    </div>
                                    <div className="font-semibold text-gray-900">Time Limit</div>
                                    <div className="text-sm text-gray-600">Per question</div>
                                </div>
                                <div className="text-center p-4 bg-green-50 rounded-xl">
                                    <div className="w-12 h-12 bg-green-500 text-white rounded-full flex items-center justify-center mx-auto mb-3">
                                        <CheckCircle className="w-6 h-6" />
                                    </div>
                                    <div className="font-semibold text-gray-900">Re-record</div>
                                    <div className="text-sm text-gray-600">Available anytime</div>
                                </div>
                            </div>

                            <ul className="space-y-4 text-gray-700 mb-8">
                                <li className="flex items-start gap-3">
                                    <div className="w-6 h-6 bg-primary-100 rounded-full flex items-center justify-center flex-shrink-0 mt-0.5">
                                        <span className="text-primary-700 font-bold text-sm">1</span>
                                    </div>
                                    <span>You'll be asked <strong>{INTERVIEW_QUESTIONS.length} targeted questions</strong> assessing different C3 soft skills</span>
                                </li>
                                <li className="flex items-start gap-3">
                                    <div className="w-6 h-6 bg-primary-100 rounded-full flex items-center justify-center flex-shrink-0 mt-0.5">
                                        <span className="text-primary-700 font-bold text-sm">2</span>
                                    </div>
                                    <span>Each question has a <strong>60-second time limit</strong>, but you can stop recording early</span>
                                </li>
                                <li className="flex items-start gap-3">
                                    <div className="w-6 h-6 bg-primary-100 rounded-full flex items-center justify-center flex-shrink-0 mt-0.5">
                                        <span className="text-primary-700 font-bold text-sm">3</span>
                                    </div>
                                    <span>You can <strong>re-record your answer</strong> if you're not satisfied with your response</span>
                                </li>
                                <li className="flex items-start gap-3">
                                    <div className="w-6 h-6 bg-primary-100 rounded-full flex items-center justify-center flex-shrink-0 mt-0.5">
                                        <span className="text-primary-700 font-bold text-sm">4</span>
                                    </div>
                                    <span><strong>Speak naturally</strong> - the AI analyzes both what you say and how you say it</span>
                                </li>
                            </ul>

                            <div className="bg-gradient-to-r from-primary-50 to-violet-50 rounded-xl p-6 mb-8 border border-primary-100">
                                <div className="flex items-start gap-3">
                                    <Sparkles className="w-5 h-5 text-primary-600 flex-shrink-0 mt-1" />
                                    <div>
                                        <h3 className="font-semibold text-gray-900 mb-1">Why This Assessment?</h3>
                                        <p className="text-sm text-gray-600 leading-relaxed">
                                            Our AI-powered platform provides
                                            <strong> objective, data-driven insights</strong> into your Curiosity, Critical Thinking,
                                            and Creativity - the soft skills that truly predict success in software engineering.
                                        </p>
                                    </div>
                                </div>
                            </div>

                            <div className="text-center">
                                <button
                                    onClick={handleStart}
                                    className="group bg-gradient-to-r from-primary-600 to-violet-600 hover:from-primary-700 hover:to-violet-700 text-white font-bold py-4 px-10 rounded-xl transition-all shadow-lg hover:shadow-xl inline-flex items-center gap-3 text-lg"
                                >
                                    <Play className="w-6 h-6 group-hover:scale-110 transition-transform" />
                                    Start Interview
                                    <ArrowRight className="w-5 h-5 group-hover:translate-x-1 transition-transform" />
                                </button>
                                <p className="text-sm text-gray-500 mt-4">
                                    ⏱️ Estimated time: {INTERVIEW_QUESTIONS.length * 2} minutes
                                </p>
                            </div>
                        </div>
                    </div>
                )}

                {step === 'question' && (
                    <div className="space-y-6 animate-fadeIn">
                        {/* Progress */}
                        <div className="text-center mb-10">
                            <div className="inline-flex items-center gap-2 px-4 py-2 bg-white rounded-full shadow-md mb-4">
                                <div className="w-2 h-2 bg-green-500 rounded-full animate-pulse"></div>
                                <span className="text-sm font-semibold text-gray-700">
                                    Question {session.currentQuestionIndex + 1} of {INTERVIEW_QUESTIONS.length}
                                </span>
                            </div>
                            <div className="w-full bg-gray-200 rounded-full h-3 max-w-md mx-auto shadow-inner">
                                <div
                                    className="bg-gradient-to-r from-primary-500 to-violet-500 h-3 rounded-full transition-all duration-500 ease-out shadow-lg"
                                    style={{ width: `${((session.currentQuestionIndex + 1) / INTERVIEW_QUESTIONS.length) * 100}%` }}
                                />
                            </div>
                            <p className="text-sm text-gray-500 mt-2">
                                {INTERVIEW_QUESTIONS.length - session.currentQuestionIndex - 1} questions remaining
                            </p>
                        </div>

                        {/* Question Card */}
                        <div className="card max-w-4xl mx-auto shadow-xl border-2 border-gray-100 hover:shadow-2xl transition-shadow">
                            <div className="mb-8">
                                <div className="flex items-center gap-3 mb-4">
                                    <div className="px-4 py-1.5 bg-gradient-to-r from-primary-100 to-violet-100 rounded-full">
                                        <span className="text-sm font-bold text-primary-700 uppercase tracking-wide">
                                            {currentQuestion.targetTrait.replace('_', ' ')}
                                        </span>
                                    </div>
                                    <div className="flex items-center gap-1.5 text-amber-500">
                                        {[...Array(5)].map((_, i) => (
                                            <Sparkles key={i} className="w-3 h-3" />
                                        ))}
                                    </div>
                                </div>
                                <h2 className="text-3xl font-bold text-gray-900 leading-relaxed">
                                    {currentQuestion.text}
                                </h2>
                            </div>

                            <div className="bg-gradient-to-br from-gray-50 to-blue-50 rounded-2xl p-8 mb-6">
                                <AudioRecorder
                                    onRecordingComplete={handleRecordingComplete}
                                    isProcessing={isProcessing}
                                    timeLimit={currentQuestion.timeLimit}
                                    onReRecord={handleReRecord}
                                    hasRecording={currentRecording !== null}
                                />
                            </div>

                            {error && (
                                <div className="mt-4 p-4 bg-red-50 border-2 border-red-200 rounded-xl flex items-center gap-3 animate-shake">
                                    <div className="w-10 h-10 bg-red-100 rounded-full flex items-center justify-center flex-shrink-0">
                                        <AlertCircle className="w-5 h-5 text-red-600" />
                                    </div>
                                    <span className="text-red-700 font-medium">{error}</span>
                                </div>
                            )}

                            {currentRecording && !isProcessing && (
                                <div className="mt-8 pt-6 border-t border-gray-200">
                                    <div className="flex items-center justify-between">
                                        <div className="flex items-center gap-2 text-green-600">
                                            <CheckCircle className="w-5 h-5" />
                                            <span className="font-medium">Response recorded</span>
                                        </div>
                                        <button
                                            onClick={handleNext}
                                            disabled={isProcessing}
                                            className="group bg-gradient-to-r from-primary-600 to-violet-600 hover:from-primary-700 hover:to-violet-700 text-white font-bold py-3 px-8 rounded-xl transition-all shadow-lg hover:shadow-xl inline-flex items-center gap-2"
                                        >
                                            {isLastQuestion ? 'Finish Interview' : 'Next Question'}
                                            <ChevronRight className="w-5 h-5 group-hover:translate-x-1 transition-transform" />
                                        </button>
                                    </div>
                                </div>
                            )}
                        </div>
                    </div>
                )}

                {step === 'processing' && (
                    <div className="text-center py-20 animate-fadeIn">
                        <div className="mb-8">
                            <div className="inline-block relative">
                                <div className="w-20 h-20 border-4 border-primary-200 border-t-primary-600 rounded-full animate-spin"></div>
                                <div className="absolute inset-0 w-20 h-20 border-4 border-transparent border-r-violet-600 rounded-full animate-spin animation-delay-150"></div>
                            </div>
                        </div>
                        <h2 className="text-3xl md:text-4xl font-bold text-gray-900 mb-3">
                            Analyzing Your Responses
                        </h2>
                        <p className="text-lg text-gray-600 mb-8">
                            Our AI is processing your voice data through 539 feature extractors
                        </p>
                        <div className="flex justify-center gap-2">
                            <div className="w-3 h-3 bg-primary-500 rounded-full animate-bounce"></div>
                            <div className="w-3 h-3 bg-violet-500 rounded-full animate-bounce animation-delay-200"></div>
                            <div className="w-3 h-3 bg-purple-500 rounded-full animate-bounce animation-delay-400"></div>
                        </div>
                    </div>
                )}

                {step === 'results' && finalResults && (
                    <div className="space-y-8">
                        <div className="card bg-green-50 border-green-200">
                            <div className="flex items-center">
                                <CheckCircle className="w-6 h-6 text-green-600 mr-3" />
                                <div>
                                    <h3 className="font-semibold text-green-900">Interview Complete!</h3>
                                    <p className="text-green-700 text-sm">Your C3 soft skills have been evaluated across all responses</p>
                                </div>
                            </div>
                        </div>

                        <div className="card">
                            <h3 className="text-2xl font-semibold mb-8 text-center text-gray-900">
                                Overall C3 Scores
                            </h3>
                            <div className="grid md:grid-cols-3 gap-8">
                                <GaugeChart value={finalResults.aggregateScores.curiosity} title="Curiosity" />
                                <GaugeChart value={finalResults.aggregateScores.critical_thinking} title="Critical Thinking" />
                                <GaugeChart value={finalResults.aggregateScores.creativity} title="Creativity" />
                            </div>
                        </div>

                        <div className="card">
                            <h3 className="text-xl font-semibold mb-4 text-gray-900">Final Recommendation</h3>
                            <RecommendationBadge recommendation={finalResults.recommendation} average={finalResults.overallAverage} />
                        </div>


                        <div className="card">
                            <h3 className="text-xl font-semibold mb-6 text-gray-900">Response Breakdown</h3>
                            <div className="space-y-6">
                                {finalResults.responses.map((response, index) => (
                                    <div key={response.questionId} className="border-b border-gray-200 last:border-0 pb-6 last:pb-0">
                                        <div className="flex items-start justify-between mb-3">
                                            <div className="flex-1">
                                                <span className="text-sm font-semibold text-gray-500">
                                                    Question {index + 1}
                                                </span>
                                                <p className="text-gray-700 mt-1">{response.questionText}</p>
                                            </div>
                                            <div className="text-right ml-4">
                                                <div className="text-2xl font-bold text-gray-900">{response.average.toFixed(1)}</div>
                                                <div className="text-xs text-gray-500">Avg Score</div>
                                            </div>
                                        </div>
                                        <div className="grid grid-cols-3 gap-4 mt-3">
                                            <ScorePill label="Curiosity" score={response.scores.curiosity} />
                                            <ScorePill label="Critical Thinking" score={response.scores.critical_thinking} />
                                            <ScorePill label="Creativity" score={response.scores.creativity} />
                                        </div>
                                        <p className="text-sm text-gray-600 mt-3 italic">"{response.transcript.substring(0, 150)}..."</p>
                                    </div>
                                ))}
                            </div>
                        </div>

                        <div className="text-center">
                            <button onClick={handleReset} className="btn-primary">
                                Start New Interview
                            </button>
                        </div>
                    </div>
                )}
            </div>
        </div>
    );
};

const ScorePill: React.FC<{ label: string; score: number }> = ({ label, score }) => (
    <div className="text-center">
        <div className="text-lg font-bold text-primary-600">{score.toFixed(1)}</div>
        <div className="text-xs text-gray-500">{label}</div>
    </div>
);

const RecommendationBadge: React.FC<{ recommendation: string; average: number }> = ({ recommendation, average }) => {
    const config = {
        STRONG_HIRE: { color: 'bg-green-100 text-green-800', text: '🌟 Strong Hire' },
        RECOMMENDED: { color: 'bg-blue-100 text-blue-800', text: '✅ Recommended' },
        CONSIDER: { color: 'bg-amber-100 text-amber-800', text: '⚠️ Consider' },
        NOT_RECOMMENDED: { color: 'bg-red-100 text-red-800', text: '❌ Not Recommended' },
    };

    const badge = config[recommendation as keyof typeof config] || config.CONSIDER;

    return (
        <div className="flex items-center justify-between">
            <span className={`px-6 py-3 rounded-lg text-lg font-semibold ${badge.color}`}>
                {badge.text}
            </span>
            <div className="text-right">
                <div className="text-3xl font-bold text-gray-900">{average.toFixed(1)}</div>
                <div className="text-sm text-gray-500">Average Score / 5</div>
            </div>
        </div>
    );
};

