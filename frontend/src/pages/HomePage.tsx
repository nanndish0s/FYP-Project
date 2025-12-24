import React from 'react';
import { Link } from 'react-router-dom';
import { Mic, Users, TrendingUp, Brain, Lightbulb, Search, Sparkles, CheckCircle2, ArrowRight, Zap } from 'lucide-react';

export const HomePage: React.FC = () => {
    return (
        <div className="min-h-screen">
            {/* Hero Section */}
            <section className="relative bg-gradient-to-br from-primary-600 via-violet-600 to-purple-700 text-white py-24 overflow-hidden">
                {/* Animated background elements */}
                <div className="absolute inset-0 opacity-20">
                    <div className="absolute top-20 left-10 w-72 h-72 bg-white rounded-full mix-blend-overlay filter blur-3xl animate-blob"></div>
                    <div className="absolute top-40 right-10 w-72 h-72 bg-purple-300 rounded-full mix-blend-overlay filter blur-3xl animate-blob animation-delay-2000"></div>
                    <div className="absolute -bottom-8 left-1/2 w-72 h-72 bg-pink-300 rounded-full mix-blend-overlay filter blur-3xl animate-blob animation-delay-4000"></div>
                </div>

                <div className="container mx-auto px-4 relative z-10">
                    <div className="max-w-4xl mx-auto text-center">
                        <div className="inline-flex items-center gap-2 px-4 py-2 bg-white/20 backdrop-blur-sm rounded-full mb-6 border border-white/30">
                            <Sparkles className="w-4 h-4" />
                            <span className="text-sm font-semibold">AI-Powered Recruitment Innovation</span>
                        </div>

                        {/* Animated Speaking Visualization */}
                        <div className="flex items-center justify-center mb-8 relative">
                            <div className="relative w-32 h-32 flex items-center justify-center">
                                {/* Breathing glow circles */}
                                <div className="absolute inset-0 bg-gradient-to-r from-yellow-300 to-pink-300 rounded-full blur-2xl animate-breathing opacity-40"></div>
                                <div className="absolute inset-0 bg-gradient-to-r from-pink-300 to-purple-300 rounded-full blur-xl animate-breathing opacity-30" style={{ animationDelay: '1s' }}></div>

                                {/* Center microphone with pulse */}
                                <div className="relative z-10 w-20 h-20 bg-gradient-to-br from-white/30 to-white/10 backdrop-blur-md rounded-full flex items-center justify-center border-2 border-white/50 shadow-2xl animate-pulse-gentle">
                                    <Mic className="w-10 h-10 text-white drop-shadow-lg" />
                                </div>
                            </div>
                        </div>

                        <h1 className="text-6xl md:text-7xl font-extrabold mb-6 leading-tight">
                            Voice-Based
                            <span className="block bg-gradient-to-r from-yellow-200 to-pink-200 bg-clip-text text-transparent">
                                Soft Skills Assessment
                            </span>
                        </h1>

                        <p className="text-xl md:text-2xl text-purple-100 mb-10 max-w-2xl mx-auto leading-relaxed">
                            Evaluate <strong>Curiosity, Critical Thinking & Creativity</strong> through advanced voice analysis with Explainable AI for transparent, fair hiring
                        </p>

                        <p className="text-sm text-purple-200/80 mb-8">
                            Project by <strong>Nanndish Satgunarjah (CB011248)</strong> | Supervisor: <strong>Dr. Tharanga Peires</strong>
                        </p>

                        <div className="flex flex-col sm:flex-row justify-center gap-4">
                            <Link
                                to="/assessment"
                                className="group bg-white text-primary-600 hover:bg-gray-50 font-bold py-4 px-8 rounded-xl transition-all shadow-lg hover:shadow-2xl hover:scale-105 inline-flex items-center justify-center gap-2"
                            >
                                <Mic className="w-5 h-5" />
                                Try Live Assessment
                                <ArrowRight className="w-5 h-5 group-hover:translate-x-1 transition-transform" />
                            </Link>
                            <Link
                                to="/candidates"
                                className="group bg-white/10 backdrop-blur-sm hover:bg-white/20 font-bold py-4 px-8 rounded-xl transition-all border-2 border-white/30 inline-flex items-center justify-center gap-2"
                            >
                                <Users className="w-5 h-5" />
                                View Demo Candidates
                            </Link>
                        </div>
                    </div>
                </div>

                {/* Wave separator */}
                <div className="absolute bottom-0 left-0 right-0">
                    <svg viewBox="0 0 1200 120" preserveAspectRatio="none" className="w-full h-16 fill-gray-50">
                        <path d="M321.39,56.44c58-10.79,114.16-30.13,172-41.86,82.39-16.72,168.19-17.73,250.45-.39C823.78,31,906.67,72,985.66,92.83c70.05,18.48,146.53,26.09,214.34,3V0H0V27.35A600.21,600.21,0,0,0,321.39,56.44Z"></path>
                    </svg>
                </div>
            </section>

            {/* Stats Section */}
            <section className="bg-gray-50 py-16">
                <div className="container mx-auto px-4">
                    <div className="grid md:grid-cols-4 gap-8 max-w-5xl mx-auto">
                        <StatCard number="44" label="Candidates Analyzed" />
                        <StatCard number="3" label="C3 Skills Assessed" />
                        <StatCard number="539" label="Voice Features Extracted" />
                        <StatCard number="100%" label="Explainable Results" />
                    </div>
                </div>
            </section>

            {/* Features Section */}
            <section className="py-20 bg-white">
                <div className="container mx-auto px-4">
                    <div className="text-center mb-16">
                        <h2 className="text-4xl md:text-5xl font-bold mb-4 text-gray-900">
                            How It Works
                        </h2>
                        <p className="text-xl text-gray-600 max-w-2xl mx-auto">
                            Our AI-powered platform analyzes voice responses through a sophisticated multi-step process
                        </p>
                    </div>

                    <div className="grid md:grid-cols-3 gap-8 max-w-6xl mx-auto">
                        <FeatureCard
                            icon={<Mic className="w-12 h-12" />}
                            title="Voice Recording"
                            description="Answer structured interview questions directly in your browser - no installations required. 60-second limit per question with re-record capability."
                            step="1"
                            gradient="from-blue-500 to-cyan-500"
                        />
                        <FeatureCard
                            icon={<Brain className="w-12 h-12" />}
                            title="AI Analysis"
                            description="Random Forest models analyze 539 acoustic and lexical features extracted from your speech patterns, tone, and content."
                            step="2"
                            gradient="from-purple-500 to-pink-500"
                        />
                        <FeatureCard
                            icon={<TrendingUp className="w-12 h-12" />}
                            title="C3 Assessment"
                            description="Get transparent scores for Curiosity, Critical Thinking, and Creativity with SHAP-powered explainability."
                            step="3"
                            gradient="from-amber-500 to-orange-500"
                        />
                    </div>
                </div>
            </section>

            {/* Skills Section */}
            <section className="bg-gradient-to-br from-gray-50 to-blue-50 py-20">
                <div className="container mx-auto px-4">
                    <div className="text-center mb-16">
                        <h2 className="text-4xl md:text-5xl font-bold mb-4 text-gray-900">
                            C3 Skills Framework
                        </h2>
                        <p className="text-xl text-gray-600 max-w-2xl mx-auto">
                            Essential cognitive soft skills for software engineering success
                        </p>
                    </div>

                    <div className="grid md:grid-cols-3 gap-8 max-w-6xl mx-auto">
                        <SkillCard
                            icon={<Search className="w-12 h-12" />}
                            title="Curiosity"
                            description="Inquisitiveness, learning drive, and desire to explore new information and technologies"
                            benefits={["Self-directed learning", "Technology exploration", "Growth mindset"]}
                            color="blue"
                        />
                        <SkillCard
                            icon={<Brain className="w-12 h-12" />}
                            title="Critical Thinking"
                            description="Analytical skills, systematic problem-solving, and evidence-based reasoning"
                            benefits={["Debugging expertise", "Architecture design", "Root cause analysis"]}
                            color="purple"
                        />
                        <SkillCard
                            icon={<Lightbulb className="w-12 h-12" />}
                            title="Creativity"
                            description="Innovation, original ideas, and novel approaches to technical challenges"
                            benefits={["Solution innovation", "Optimization", "Novel thinking"]}
                            color="amber"
                        />
                    </div>
                </div>
            </section>

            {/* Why XAI Section */}
            <section className="py-20 bg-white">
                <div className="container mx-auto px-4">
                    <div className="max-w-5xl mx-auto">
                        <div className="grid md:grid-cols-2 gap-12 items-center">
                            <div>
                                <div className="inline-flex items-center gap-2 px-4 py-2 bg-primary-50 text-primary-700 rounded-full mb-4">
                                    <Zap className="w-4 h-4" />
                                    <span className="text-sm font-semibold">Transparency First</span>
                                </div>
                                <h2 className="text-4xl font-bold mb-6 text-gray-900">
                                    Why Explainable AI Matters
                                </h2>
                                <p className="text-lg text-gray-600 mb-6 leading-relaxed">
                                    Traditional AI hiring tools are "black boxes" that hide biases and lack accountability. Our XAI framework uses SHAP (SHapley Additive exPlanations) to provide transparent, interpretable results.
                                </p>
                                <div className="space-y-3">
                                    <BenefitItem text="Fair and unbiased decision making" />
                                    <BenefitItem text="Actionable feedback for candidates" />
                                    <BenefitItem text="Regulatory compliance ready" />
                                    <BenefitItem text="Trust through transparency" />
                                </div>
                            </div>
                            <div className="relative">
                                <div className="bg-gradient-to-br from-primary-100 to-violet-100 rounded-2xl p-8 shadow-xl">
                                    <div className="bg-white rounded-xl p-6 shadow-lg">
                                        <h3 className="font-bold text-gray-900 mb-4">Research Foundation</h3>
                                        <ul className="space-y-2 text-sm text-gray-600">
                                            <li className="flex items-start gap-2">
                                                <CheckCircle2 className="w-5 h-5 text-green-500 flex-shrink-0 mt-0.5" />
                                                <span>Built on peer-reviewed research (2025)</span>
                                            </li>
                                            <li className="flex items-start gap-2">
                                                <CheckCircle2 className="w-5 h-5 text-green-500 flex-shrink-0 mt-0.5" />
                                                <span>Validates C3 skills from voice data</span>
                                            </li>
                                            <li className="flex items-start gap-2">
                                                <CheckCircle2 className="w-5 h-5 text-green-500 flex-shrink-0 mt-0.5" />
                                                <span>First integrated voice-based XAI framework</span>
                                            </li>
                                        </ul>
                                        <Link to="/research" className="mt-6 inline-flex items-center gap-2 text-primary-600 font-semibold hover:gap-3 transition-all">
                                            View Research Evidence
                                            <ArrowRight className="w-4 h-4" />
                                        </Link>
                                    </div>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
            </section>

            {/* CTA Section */}
            <section className="py-20 bg-gradient-to-r from-primary-600 to-violet-600 text-white relative overflow-hidden">
                <div className="absolute inset-0 opacity-10">
                    <div className="absolute top-0 left-0 w-96 h-96 bg-white rounded-full filter blur-3xl"></div>
                    <div className="absolute bottom-0 right-0 w-96 h-96 bg-pink-300 rounded-full filter blur-3xl"></div>
                </div>

                <div className="container mx-auto px-4 text-center relative z-10">
                    <h2 className="text-4xl md:text-5xl font-bold mb-4">Ready to Get Started?</h2>
                    <p className="text-xl text-purple-100 mb-10 max-w-2xl mx-auto">
                        Experience transparent, AI-powered soft skills assessment in action
                    </p>
                    <div className="flex flex-col sm:flex-row justify-center gap-4">
                        <Link to="/assessment" className="group bg-white text-primary-600 hover:bg-gray-50 font-bold py-4 px-10 rounded-xl transition-all shadow-lg hover:shadow-2xl inline-flex items-center justify-center gap-2">
                            <Mic className="w-5 h-5" />
                            Start Live Assessment
                            <ArrowRight className="w-5 h-5 group-hover:translate-x-1 transition-transform" />
                        </Link>
                        <Link to="/candidates" className="group bg-white/10 backdrop-blur-sm hover:bg-white/20 font-bold py-4 px-10 rounded-xl transition-all border-2 border-white/30 inline-flex items-center justify-center gap-2">
                            <Users className="w-5 h-5" />
                            Browse 44 Demo Candidates
                        </Link>
                    </div>
                </div>
            </section>
        </div>
    );
};

// Component Definitions
const StatCard: React.FC<{ number: string; label: string }> = ({ number, label }) => (
    <div className="text-center">
        <div className="text-5xl font-extrabold text-primary-600 mb-2">{number}</div>
        <div className="text-gray-600 font-medium">{label}</div>
    </div>
);

interface FeatureCardProps {
    icon: React.ReactNode;
    title: string;
    description: string;
    step: string;
    gradient: string;
}

const FeatureCard: React.FC<FeatureCardProps> = ({ icon, title, description, step, gradient }) => (
    <div className="group relative bg-white rounded-2xl p-8 shadow-lg hover:shadow-2xl transition-all duration-300 hover:-translate-y-2 border border-gray-100">
        <div className="absolute -top-4 -left-4 w-12 h-12 bg-gradient-to-br ${gradient} rounded-xl flex items-center justify-center text-white font-bold text-lg shadow-lg">
            {step}
        </div>
        <div className={`text-white mb-6 w-16 h-16 bg-gradient-to-br ${gradient} rounded-xl flex items-center justify-center group-hover:scale-110 transition-transform`}>
            {icon}
        </div>
        <h3 className="text-2xl font-bold mb-3 text-gray-900">{title}</h3>
        <p className="text-gray-600 leading-relaxed">{description}</p>
    </div>
);

interface SkillCardProps {
    icon: React.ReactNode;
    title: string;
    description: string;
    benefits: string[];
    color: string;
}

const SkillCard: React.FC<SkillCardProps> = ({ icon, title, description, benefits, color }) => {
    const colorClasses = {
        blue: 'from-blue-500 to-cyan-500',
        purple: 'from-purple-500 to-pink-500',
        amber: 'from-amber-500 to-orange-500'
    };

    return (
        <div className="group bg-white rounded-2xl p-8 shadow-lg hover:shadow-2xl transition-all duration-300 hover:-translate-y-2 border border-gray-100">
            <div className={`w- 16 h-16 bg-gradient-to-br ${colorClasses[color as keyof typeof colorClasses]} rounded-xl flex items-center justify-center text-white mb-6 group-hover:scale-110 transition-transform`}>
                {icon}
            </div>
            <h3 className="text-2xl font-bold mb-3 text-gray-900">{title}</h3>
            <p className="text-gray-600 mb-4 leading-relaxed">{description}</p>
            <div className="space-y-2">
                {benefits.map((benefit, index) => (
                    <div key={index} className="flex items-center gap-2 text-sm text-gray-700">
                        <CheckCircle2 className="w-4 h-4 text-green-500 flex-shrink-0" />
                        <span>{benefit}</span>
                    </div>
                ))}
            </div>
        </div>
    );
};

const BenefitItem: React.FC<{ text: string }> = ({ text }) => (
    <div className="flex items-center gap-3">
        <div className="w-6 h-6 bg-green-100 rounded-full flex items-center justify-center flex-shrink-0">
            <CheckCircle2 className="w-4 h-4 text-green-600" />
        </div>
        <span className="text-gray-700">{text}</span>
    </div>
);
