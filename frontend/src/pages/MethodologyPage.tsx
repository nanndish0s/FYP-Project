import React from 'react';
import { Mic, Activity, Brain, BarChart, ArrowRight, Settings, Sparkles, Zap, Target } from 'lucide-react';

export const MethodologyPage: React.FC = () => {
    return (
        <div className="min-h-screen bg-gradient-to-br from-gray-50 to-blue-50 py-12">
            <div className="container mx-auto px-4 max-w-6xl">
                {/* Header */}
                <div className="text-center mb-16">
                    <div className="inline-flex items-center gap-2 px-4 py-2 bg-primary-100 text-primary-700 rounded-full mb-6 shadow-sm">
                        <Target className="w-4 h-4" />
                        <span className="text-sm font-semibold">AI Pipeline Overview</span>
                    </div>

                    <h1 className="text-5xl md:text-6xl font-extrabold mb-6">
                        <span className="text-gray-900">How It</span>
                        <span className="block bg-gradient-to-r from-primary-600 to-violet-600 bg-clip-text text-transparent">
                            Works
                        </span>
                    </h1>

                    <p className="text-xl md:text-2xl text-gray-600 max-w-3xl mx-auto leading-relaxed">
                        Our <strong className="text-primary-600">Transparent AI Pipeline</strong> for Soft Skills Assessment
                    </p>
                </div>

                {/* Pipeline Overview */}
                <div className="mb-20">
                    <div className="relative">
                        {/* Connecting Line */}
                        <div className="hidden lg:block absolute top-1/2 left-0 w-full h-1 bg-gradient-to-r from-blue-200 via-purple-200 to-emerald-200 transform -translate-y-1/2" style={{ top: '80px', zIndex: 0 }}></div>

                        <div className="grid md:grid-cols-2 lg:grid-cols-4 gap-6 relative z-10">
                            <StepCard
                                icon={<Mic className="w-8 h-8" />}
                                title="1. Input"
                                description="Voice Recording"
                                gradient="from-blue-500 to-cyan-500"
                                delay="0"
                            />
                            <StepCard
                                icon={<Activity className="w-8 h-8" />}
                                title="2. Analysis"
                                description="Feature Extraction (539)"
                                gradient="from-indigo-500 to-purple-500"
                                delay="100"
                            />
                            <StepCard
                                icon={<Brain className="w-8 h-8" />}
                                title="3. Prediction"
                                description="Random Forest Models"
                                gradient="from-purple-500 to-pink-500"
                                delay="200"
                            />
                            <StepCard
                                icon={<BarChart className="w-8 h-8" />}
                                title="4. Result"
                                description="C3 Scores + SHAP XAI"
                                gradient="from-emerald-500 to-green-500"
                                delay="300"
                            />
                        </div>
                    </div>
                </div>

                {/* Detailed Steps */}
                <div className="space-y-8">
                    {/* Step 1 */}
                    <DetailedStep
                        number="1"
                        title="Audio Processing & Transcription"
                        icon={<Mic className="w-7 h-7" />}
                        iconGradient="from-blue-500 to-cyan-500"
                        content={
                            <div className="space-y-4">
                                <div className="grid md:grid-cols-3 gap-4 mb-6">
                                    <InfoPill label="Input Format" value="WAV, 16kHz Mono" />
                                    <InfoPill label="Duration" value="30-90 seconds" />
                                    <InfoPill label="Engine" value="OpenAI Whisper" />
                                </div>
                                <ul className="space-y-3 text-gray-700">
                                    <li className="flex items-start gap-3">
                                        <div className="w-6 h-6 bg-blue-100 rounded-full flex items-center justify-center flex-shrink-0 mt-0.5">
                                            <span className="text-blue-700 font-bold text-sm">•</span>
                                        </div>
                                        <span><strong>Input:</strong> Candidates record responses using the Web Audio API (browser-based, no installation)</span>
                                    </li>
                                    <li className="flex items-start gap-3">
                                        <div className="w-6 h-6 bg-blue-100 rounded-full flex items-center justify-center flex-shrink-0 mt-0.5">
                                            <span className="text-blue-700 font-bold text-sm">•</span>
                                        </div>
                                        <span><strong>Transcription:</strong> Whisper converts speech to text with high accuracy across accents and languages</span>
                                    </li>
                                    <li className="flex items-start gap-3">
                                        <div className="w-6 h-6 bg-blue-100 rounded-full flex items-center justify-center flex-shrink-0 mt-0.5">
                                            <span className="text-blue-700 font-bold text-sm">•</span>
                                        </div>
                                        <span><strong>Output:</strong> Both the original audio waveform and transcript are used for dual-modality analysis</span>
                                    </li>
                                </ul>
                            </div>
                        }
                    />

                    {/* Step 2 */}
                    <DetailedStep
                        number="2"
                        title="Dual-Modality Feature Engineering"
                        icon={<Settings className="w-7 h-7" />}
                        iconGradient="from-indigo-500 to-purple-500"
                        content={
                            <div>
                                <div className="bg-gradient-to-r from-purple-50 to-blue-50 rounded-xl p-6 mb-6 border border-purple-100">
                                    <div className="flex items-start gap-3 mb-4">
                                        <Sparkles className="w-5 h-5 text-purple-600 flex-shrink-0 mt-1" />
                                        <div>
                                            <h4 className="font-bold text-gray-900 mb-2">539 Total Features Extracted</h4>
                                            <p className="text-sm text-gray-600">Combining acoustic (how you speak) and lexical (what you say) signals for comprehensive assessment</p>
                                        </div>
                                    </div>
                                </div>

                                <div className="grid md:grid-cols-2 gap-6">
                                    <FeatureBox
                                        title="Acoustic Features"
                                        subtitle="How You Speak"
                                        gradient="from-indigo-500 to-purple-500"
                                        features={[
                                            'Pitch (F0) & Variation',
                                            'Jitter & Shimmer (Voice Quality)',
                                            'Speaking Rate & Pauses',
                                            'Energy & Loudness',
                                            'Mel-Frequency Cepstral Coefficients (MFCC)'
                                        ]}
                                        tool="Librosa/COVAREP"
                                    />
                                    <FeatureBox
                                        title="Lexical Features"
                                        subtitle="What You Say"
                                        gradient="from-blue-500 to-cyan-500"
                                        features={[
                                            'Word Count & Complexity',
                                            'Filler Words (um, uh, like)',
                                            'Sentiment & Tone Analysis',
                                            'Vocabulary Richness (TTR)',
                                            'Semantic Coherence'
                                        ]}
                                        tool="NLP Analysis"
                                    />
                                </div>
                            </div>
                        }
                    />

                    {/* Step 3 */}
                    <DetailedStep
                        number="3"
                        title="Machine Learning Models"
                        icon={<Brain className="w-7 h-7" />}
                        iconGradient="from-purple-500 to-pink-500"
                        content={
                            <div className="space-y-6">
                                <div className="grid md:grid-cols-3 gap-4">
                                    <ModelBadge color="blue" label="Curiosity Model" />
                                    <ModelBadge color="purple" label="Critical Thinking Model" />
                                    <ModelBadge color="amber" label="Creativity Model" />
                                </div>

                                <ul className="space-y-3 text-gray-700">
                                    <li className="flex items-start gap-3">
                                        <Zap className="w-5 h-5 text-purple-600 flex-shrink-0 mt-0.5" />
                                        <span><strong>Algorithm:</strong> Random Forest Regressors (ensemble of decision trees for robust predictions)</span>
                                    </li>
                                    <li className="flex items-start gap-3">
                                        <Zap className="w-5 h-5 text-purple-600 flex-shrink-0 mt-0.5" />
                                        <span><strong>Training Data:</strong> 44 labeled interview samples from CMU-MOSEI dataset, annotated with C3 scores using Llama 3.3-70B</span>
                                    </li>
                                    <li className="flex items-start gap-3">
                                        <Zap className="w-5 h-5 text-purple-600 flex-shrink-0 mt-0.5" />
                                        <span><strong>Approach:</strong> Each C3 trait has a dedicated model trained on the full 539-feature vector</span>
                                    </li>
                                </ul>
                            </div>
                        }
                    />

                    {/* Step 4 */}
                    <DetailedStep
                        number="4"
                        title="Explainable AI (XAI) with SHAP"
                        icon={<BarChart className="w-7 h-7" />}
                        iconGradient="from-emerald-500 to-green-500"
                        content={
                            <div className="bg-gradient-to-br from-emerald-50 to-green-50 p-8 rounded-2xl border-2 border-emerald-200">
                                <div className="mb-6">
                                    <h4 className="text-2xl font-bold text-emerald-900 mb-3 flex items-center gap-2">
                                        <Sparkles className="w-6 h-6" />
                                        SHAP (SHapley Additive exPlanations)
                                    </h4>
                                    <p className="text-gray-800 text-lg mb-6">
                                        We use game theory-based SHAP values to make the "black box" transparent, showing exactly which features contributed to each score.
                                    </p>
                                </div>

                                <div className="grid md:grid-cols-2 gap-6">
                                    <div className="bg-white rounded-xl p-6 shadow-lg">
                                        <h5 className="font-bold text-emerald-900 mb-4 flex items-center gap-2">
                                            <div className="w-8 h-8 bg-emerald-100 rounded-full flex items-center justify-center">
                                                ✓
                                            </div>
                                            Transparency Benefits
                                        </h5>
                                        <ul className="space-y-3 text-sm text-gray-700">
                                            <li className="flex items-start gap-2">
                                                <span className="text-emerald-600 font-bold">→</span>
                                                <span>Know exactly <em>why</em> a score was given</span>
                                            </li>
                                            <li className="flex items-start gap-2">
                                                <span className="text-emerald-600 font-bold">→</span>
                                                <span>See which features (pitch, speed, word choice) influenced the result most</span>
                                            </li>
                                            <li className="flex items-start gap-2">
                                                <span className="text-emerald-600 font-bold">→</span>
                                                <span>Identify and remove potential biases in decision-making</span>
                                            </li>
                                        </ul>
                                    </div>

                                    <div className="bg-white rounded-xl p-6 shadow-lg">
                                        <h5 className="font-bold text-emerald-900 mb-4 flex items-center gap-2">
                                            <div className="w-8 h-8 bg-emerald-100 rounded-full flex items-center justify-center">
                                                ⚖️
                                            </div>
                                            Fairness Assurance
                                        </h5>
                                        <ul className="space-y-3 text-sm text-gray-700">
                                            <li className="flex items-start gap-2">
                                                <span className="text-emerald-600 font-bold">→</span>
                                                <span>Ensures decisions aren't based on discriminatory factors (e.g., accent, demographic markers)</span>
                                            </li>
                                            <li className="flex items-start gap-2">
                                                <span className="text-emerald-600 font-bold">→</span>
                                                <span>Enables auditing and regulatory compliance (GDPR, AI Act)</span>
                                            </li>
                                            <li className="flex items-start gap-2">
                                                <span className="text-emerald-600 font-bold">→</span>
                                                <span>Builds trust with candidates and recruiters</span>
                                            </li>
                                        </ul>
                                    </div>
                                </div>
                            </div>
                        }
                    />
                </div>
            </div>
        </div>
    );
};

// Component Definitions
interface StepCardProps {
    icon: React.ReactNode;
    title: string;
    description: string;
    gradient: string;
    delay: string;
}

const StepCard: React.FC<StepCardProps> = ({ icon, title, description, gradient, delay }) => (
    <div
        className="group bg-white p-6 rounded-2xl shadow-lg hover:shadow-2xl border-2 border-gray-100 text-center relative transition-all duration-300 hover:-translate-y-2 animate-fadeIn"
        style={{ animationDelay: `${delay}ms` }}
    >
        <div className={`w-20 h-20 bg-gradient-to-br ${gradient} rounded-2xl flex items-center justify-center mx-auto mb-4 shadow-xl text-white group-hover:scale-110 transition-transform`}>
            {icon}
        </div>
        <h3 className="font-bold text-gray-900 mb-2 text-lg">{title}</h3>
        <p className="text-gray-600 font-medium text-sm">{description}</p>
    </div>
);

interface DetailedStepProps {
    number: string;
    title: string;
    icon: React.ReactNode;
    iconGradient: string;
    content: React.ReactNode;
}

const DetailedStep: React.FC<DetailedStepProps> = ({ number, title, icon, iconGradient, content }) => (
    <div className="bg-white p-8 md:p-10 rounded-2xl shadow-xl border-2 border-gray-100 hover:shadow-2xl transition-shadow">
        <div className="flex flex-col md:flex-row gap-6 mb-6">
            <div className="flex-shrink-0">
                <div className={`w-16 h-16 bg-gradient-to-br ${iconGradient} rounded-2xl flex items-center justify-center text-white font-bold text-2xl shadow-lg`}>
                    {number}
                </div>
            </div>
            <div className="flex-1">
                <h3 className="text-3xl font-bold text-gray-900 flex items-center gap-3">
                    <span className={`text-transparent bg-clip-text bg-gradient-to-r ${iconGradient}`}>{title}</span>
                </h3>
            </div>
        </div>
        {content}
    </div>
);

const InfoPill: React.FC<{ label: string; value: string }> = ({ label, value }) => (
    <div className="bg-blue-50 rounded-lg p-4 text-center border border-blue-100">
        <div className="text-xs font-semibold text-blue-600 uppercase tracking-wide mb-1">{label}</div>
        <div className="font-bold text-gray-900">{value}</div>
    </div>
);

interface FeatureBoxProps {
    title: string;
    subtitle: string;
    gradient: string;
    features: string[];
    tool: string;
}

const FeatureBox: React.FC<FeatureBoxProps> = ({ title, subtitle, gradient, features, tool }) => (
    <div className="bg-white rounded-xl p-6 border-2 border-gray-100 shadow-lg hover:shadow-xl transition-shadow">
        <div className={`inline-block px-4 py-2 bg-gradient-to-r ${gradient} rounded-lg text-white font-bold mb-4`}>
            {title}
        </div>
        <p className="text-sm text-gray-500 mb-4">{subtitle}</p>
        <ul className="space-y-2.5 mb-4">
            {features.map((feature, index) => (
                <li key={index} className="flex items-start gap-2 text-sm text-gray-700">
                    <span className="text-primary-600 font-bold mt-0.5">•</span>
                    <span>{feature}</span>
                </li>
            ))}
        </ul>
        <div className="pt-4 border-t border-gray-100">
            <span className="inline-block px-3 py-1 bg-gray-100 rounded-full text-xs font-semibold text-gray-600">
                {tool}
            </span>
        </div>
    </div>
);

const ModelBadge: React.FC<{ color: string; label: string }> = ({ color, label }) => {
    const colors = {
        blue: 'bg-blue-100 text-blue-700 border-blue-200',
        purple: 'bg-purple-100 text-purple-700 border-purple-200',
        amber: 'bg-amber-100 text-amber-700 border-amber-200'
    };

    return (
        <div className={`${colors[color as keyof typeof colors]} border-2 rounded-xl p-4 text-center font-bold shadow-md`}>
            {label}
        </div>
    );
};
