import React, { useState } from 'react';
import { BookOpen, ChevronDown, ChevronUp, Target, Lightbulb, AlertTriangle, Award, FileText, Sparkles } from 'lucide-react';

export const ResearchPage: React.FC = () => {
    const [expandedSection, setExpandedSection] = useState<string | null>('background');

    const toggleSection = (section: string) => {
        setExpandedSection(expandedSection === section ? null : section);
    };

    return (
        <div className="min-h-screen bg-gradient-to-br from-gray-50 to-blue-50 py-12">
            <div className="container mx-auto px-4 max-w-6xl">
                {/* Header */}
                <div className="text-center mb-10">
                    <div className="inline-flex items-center gap-2 px-3 py-1.5 bg-primary-100 text-primary-700 rounded-full mb-4 shadow-sm">
                        <BookOpen className="w-3.5 h-3.5" />
                        <span className="text-xs font-semibold">Academic Foundation</span>
                    </div>

                    <h1 className="text-4xl md:text-5xl font-extrabold mb-4">
                        <span className="bg-gradient-to-r from-primary-600 to-violet-600 bg-clip-text text-transparent">
                            Research
                        </span>
                        <span className="block text-gray-900 mt-1">Foundation</span>
                    </h1>

                    <p className="text-lg text-gray-600 max-w-3xl mx-auto leading-relaxed">
                        Literature evidence demonstrating the <strong className="text-primary-600">novelty</strong> and <strong className="text-primary-600">academic rigor</strong> of this project
                    </p>
                </div>

                {/* Background & Problem Domain */}
                <CollapsibleSection
                    title="Background & Problem Domain"
                    icon={<Target className="w-6 h-6" />}
                    iconGradient="from-blue-500 to-cyan-500"
                    isExpanded={expandedSection === 'background'}
                    onToggle={() => toggleSection('background')}
                >
                    <p className="text-gray-700 leading-relaxed mb-4">
                        In the highly competitive field of Software Engineering, organisations increasingly recognize that a candidate's technical ability is only one part of the equation for success. Higher-order soft skills – such as <strong className="text-primary-600">Critical Thinking, Curiosity, and Creativity (the 'C3' skills)</strong> - are essential for innovation, problem-solving, and adaptability.
                    </p>
                    <p className="text-gray-700 leading-relaxed mb-4">
                        However, traditional recruitment methods like CV screening and unstructured interviews are subjective, inefficient, and prone to human biases. To fix these problems, companies are quickly starting to use Artificial Intelligence (AI) to automate hiring and make it more consistent.
                    </p>
                    <div className="bg-gradient-to-r from-amber-50 to-orange-50 border-2 border-amber-200 rounded-xl p-4 shadow-lg">
                        <div className="flex items-start gap-3">
                            <div className="w-8 h-8 bg-amber-100 rounded-full flex items-center justify-center flex-shrink-0">
                                <Sparkles className="w-4 h-4 text-amber-600" />
                            </div>
                            <div>
                                <h4 className="font-semibold text-gray-900 mb-2 text-sm">The Problem</h4>
                                <p className="text-gray-700 leading-relaxed text-sm">
                                    Many AI systems have created their own serious issues. They are like "black boxes" - you can't see how they're making decisions, which can hide or even worsen unfair biases. Furthermore, most current systems focus on text-based resume analysis or basic skills like "confidence" rather than the complex, cognitive C3 skills.
                                </p>
                            </div>
                        </div>
                    </div>
                </CollapsibleSection>

                {/* Problem Statement */}
                <CollapsibleSection
                    title="Problem Statement"
                    icon={<AlertTriangle className="w-6 h-6" />}
                    iconGradient="from-red-500 to-pink-500"
                    isExpanded={expandedSection === 'problem'}
                    onToggle={() => toggleSection('problem')}
                >
                    <div className="bg-gradient-to-br from-red-50 to-pink-50 border-l-4 border-red-500 p-5 rounded-xl shadow-lg">
                        <p className="text-gray-800 font-medium leading-relaxed">
                            There is a <strong className="text-red-700">lack of a transparent and validated framework</strong> to automatically assess the higher-order soft skills like Curiosity, Critical Thinking, and Creativity from voice data using Explainable AI (XAI) within the context of software engineering recruitment.
                        </p>
                    </div>
                </CollapsibleSection>

                {/* Key Related Work */}
                <CollapsibleSection
                    title="Key Related Work"
                    icon={<FileText className="w-6 h-6" />}
                    iconGradient="from-indigo-500 to-purple-500"
                    isExpanded={expandedSection === 'related'}
                    onToggle={() => toggleSection('related')}
                >
                    <div className="space-y-6">
                        <div className="bg-gradient-to-r from-indigo-50 to-purple-50 rounded-xl p-4 border border-indigo-200 mb-4">
                            <p className="text-gray-700 leading-relaxed text-sm">
                                <strong className="text-indigo-900">5 peer-reviewed studies (2025)</strong> demonstrate that while individual C3 skills can be detected from voice, <strong>no integrated framework exists</strong> for all three skills with explainable AI in recruitment contexts.
                            </p>
                        </div>

                        <RelatedWorkCard
                            authors="Rajbhar, S., et al. (2025)"
                            contribution="Proposed a multimodal (audio, visual, text) mock interview platform to evaluate communication, emotion, and confidence."
                            limitation="Limited adaptation for specialised job roles; no discussion of explainability."
                            gap="Fails to assess higher-order C3 skills; lacks XAI for transparency."
                            relevance="Demonstrates the feasibility of multimodal assessment but highlights the exact gap this project targets: C3 skill evaluation and explainability."
                        />
                        <RelatedWorkCard
                            authors="Singla, L., et al. (2025)"
                            contribution="Developed a hybrid Transformer system with an XAI module (SHAP) for text-based resume screening and candidate-job matching."
                            limitation="Lacks voice or multimodal assessment; XAI is only applied to text-based decisions."
                            gap="Does not assess C3 skills; XAI for voice-based soft skills remains unexplored."
                            relevance="Shows it's possible to use XAI in hiring, but only with text, highlighting the need for XAI that works with voice."
                        />
                        <RelatedWorkCard
                            authors="Guerrero-Sosa, J.D.T., et al. (2025)"
                            contribution="Presented a multimodal platform (audio, video, text) to assess skills, including Creativity, using fuzzy logic to explain the scores."
                            limitation="Small sample size (N=49); fairness issues related to cultural/neurodiversity; does not assess Curiosity or Critical Thinking."
                            gap="Assesses one C3 skill but not all three; fuzzy logic is interpretable but different from model-agnostic XAI (SHAP/LIME)."
                            relevance="Key competitor, but its limited scope (1 of 3 C3 skills) and unaddressed fairness challenges justify this project's more comprehensive framework."
                        />
                        <RelatedWorkCard
                            authors="Ebrahimpourlighvani, A. (2025)"
                            contribution="Developed a test showing that Critical Thinking (CT) is measurable from oral responses and is statistically correlated with vocal features."
                            limitation="Small sample (N=31); focused on an L2 academic context, not recruitment; no XAI."
                            gap="Lacks C3 integration, recruitment focus, and explainability."
                            relevance="Provides crucial evidence that a core C3 skill (Critical Thinking) is measurably linked to vocal features."
                        />
                        <RelatedWorkCard
                            authors="Shoukat, S., et al. (2025)"
                            contribution="Empirically linked measurable vocal features (pitch range, intonation contours) to student curiosity and cognitive engagement."
                            limitation="Small sample (8 teachers); non-recruitment context (classrooms)."
                            gap="Lacks computational/AI-based modeling for automated curiosity detection in a professional context."
                            relevance="Provides crucial evidence that a core C3 skill (Curiosity) is measurably linked to prosodic/vocal features."
                        />
                    </div>
                </CollapsibleSection>

                {/* Research Gaps */}
                <CollapsibleSection
                    title="Identified Research Gaps"
                    icon={<Lightbulb className="w-6 h-6" />}
                    iconGradient="from-amber-500 to-orange-500"
                    isExpanded={expandedSection === 'gaps'}
                    onToggle={() => toggleSection('gaps')}
                >
                    <div className="grid md:grid-cols-3 gap-6">
                        <GapCard
                            number={1}
                            title="Assessment Focus Gap"
                            subtitle="Basic vs. Higher-Order Skills"
                            description="The vast majority of AI-driven recruitment systems focus on assessing basic communication skills, confidence, or personality traits. There is a clear lack of systems that assess the higher-order cognitive soft skills – Critical Thinking, Curiosity, and Creativity."
                        />
                        <GapCard
                            number={2}
                            title="Modality Gap"
                            subtitle="Text/Video vs. Integrated Voice Framework"
                            description="Existing systems are heavily skewed towards text-based resume screening or full multimodal video analysis. No integrated framework exists to assess all three C3 skills simultaneously from voice data in a recruitment setting."
                        />
                        <GapCard
                            number={3}
                            title="Transparency Gap"
                            subtitle="Black Box vs. Explainable AI"
                            description="The most significant gap. Existing systems are consistently criticized as opaque 'black boxes'. A framework that provides XAI for voice-based soft skills assessment remains critical and unexplored."
                            isHighlighted={true}
                        />
                    </div>
                </CollapsibleSection>

                {/* Contribution */}
                <CollapsibleSection
                    title="Project Contribution"
                    icon={<Award className="w-6 h-6" />}
                    iconGradient="from-green-500 to-emerald-500"
                    isExpanded={expandedSection === 'contribution'}
                    onToggle={() => toggleSection('contribution')}
                >
                    <div className="grid md:grid-cols-3 gap-6">
                        <ContributionCard
                            title="Practical Impact"
                            subtitle="Problem Domain"
                            description="First proof-of-concept for a tool that provides recruiters with transparent, evidence-based insights into a candidate's cognitive abilities. This can lead to fairer, more accurate, and less biased hiring decisions."
                            gradient="from-blue-500 to-cyan-500"
                            icon="💼"
                        />
                        <ContributionCard
                            title="Algorithmic Novelty"
                            subtitle="Research Domain"
                            description="Design and development of a novel, integrated, explainable AI framework specifically for assessing cognitive skills from voice. This involves combining acoustic, lexical, and semantic features with XAI tools that give clear, practical feedback."
                            gradient="from-green-500 to-emerald-500"
                            icon="🔬"
                        />
                        <ContributionCard
                            title="Academic Advancement"
                            subtitle="Body of Knowledge"
                            description="First to bridge the gap between multimodal interaction analysis and explainable AI for cognitive skill assessment in recruitment. Provides a validated methodology and new benchmark for future research."
                            gradient="from-purple-500 to-pink-500"
                            icon="📚"
                        />
                    </div>
                </CollapsibleSection>

                {/* Research Questions */}
                <CollapsibleSection
                    title="Research Questions & Objectives"
                    icon={<Target className="w-6 h-6" />}
                    iconGradient="from-violet-500 to-purple-500"
                    isExpanded={expandedSection === 'questions'}
                    onToggle={() => toggleSection('questions')}
                >
                    <div className="space-y-8">
                        <div>
                            <div className="bg-gradient-to-r from-primary-500 to-violet-500 rounded-xl p-8 text-white shadow-xl mb-6">
                                <h3 className="text-2xl font-bold mb-4 flex items-center gap-2">
                                    <Target className="w-6 h-6" />
                                    Research Aim
                                </h3>
                                <p className="text-lg leading-relaxed">
                                    To design, develop, and validate a novel explainable AI framework capable of assessing a candidate's Curiosity, Critical Thinking, and Creativity from their spoken responses during a job interview.
                                </p>
                            </div>
                        </div>

                        <div>
                            <h3 className="text-2xl font-bold text-gray-900 mb-6">Research Questions</h3>
                            <div className="space-y-4">
                                <ResearchQuestion
                                    number="RQ1"
                                    question="To what extent can features extracted from a candidate's spoken responses and interaction patterns be used to reliably model and predict their levels of Curiosity, Critical Thinking, and Creativity?"
                                />
                                <ResearchQuestion
                                    number="RQ2"
                                    question="What explainable AI methodologies are most effective for generating transparent, fair, and human-interpretable explanations for AI-driven assessments of cognitive soft skills in a recruitment context?"
                                />
                                <ResearchQuestion
                                    number="RQ3"
                                    question="How can the performance, fairness, and utility of an integrated, voice-based XAI framework for C3 skill assessment be rigorously evaluated against both quantitative benchmarks and qualitative human expert judgment?"
                                />
                            </div>
                        </div>
                    </div>
                </CollapsibleSection>
            </div>
        </div>
    );
};

// Component Definitions
interface CollapsibleSectionProps {
    title: string;
    icon: React.ReactNode;
    iconGradient: string;
    isExpanded: boolean;
    onToggle: () => void;
    children: React.ReactNode;
}

const CollapsibleSection: React.FC<CollapsibleSectionProps> = ({ title, icon, iconGradient, isExpanded, onToggle, children }) => (
    <div className="mb-6">
        <button
            onClick={onToggle}
            className="w-full bg-white rounded-xl shadow-lg border-2 border-gray-100 hover:shadow-xl transition-all p-4 text-left group"
        >
            <div className="flex items-center justify-between">
                <div className="flex items-center gap-3 flex-1">
                    <div className={`p-3 bg-gradient-to-br ${iconGradient} rounded-lg text-white shadow-lg group-hover:scale-110 transition-transform`}>
                        {icon}
                    </div>
                    <h2 className="text-2xl font-bold text-gray-900">{title}</h2>
                </div>
                <div className="ml-4">
                    {isExpanded ? (
                        <ChevronUp className="w-6 h-6 text-gray-400" />
                    ) : (
                        <ChevronDown className="w-6 h-6 text-gray-400" />
                    )}
                </div>
            </div>
        </button>
        {isExpanded && (
            <div className="mt-4 bg-white rounded-xl shadow-lg border-2 border-gray-100 p-6 animate-fadeIn">
                {children}
            </div>
        )}
    </div>
);

interface RelatedWorkCardProps {
    authors: string;
    contribution: string;
    limitation: string;
    gap: string;
    relevance: string;
}

const RelatedWorkCard: React.FC<RelatedWorkCardProps> = ({ authors, contribution, limitation, gap, relevance }) => (
    <div className="bg-white border-2 border-gray-200 rounded-xl p-5 hover:shadow-xl transition-all hover:border-primary-300">
        <div className="flex items-center gap-2 mb-3">
            <div className="px-2.5 py-0.5 bg-primary-100 rounded-full">
                <span className="text-xs font-bold text-primary-700 uppercase tracking-wide">Study</span>
            </div>
            <h4 className="font-bold text-primary-600">{authors}</h4>
        </div>
        <div className="space-y-2.5 text-sm">
            <div className="pl-4 border-l-2 border-blue-200">
                <span className="font-bold text-blue-700 block mb-1">Contribution</span>
                <p className="text-gray-700 leading-relaxed">{contribution}</p>
            </div>
            <div className="pl-4 border-l-2 border-red-200">
                <span className="font-bold text-red-700 block mb-1">Limitation</span>
                <p className="text-gray-700 leading-relaxed">{limitation}</p>
            </div>
            <div className="pl-4 border-l-2 border-amber-200">
                <span className="font-bold text-amber-700 block mb-1">Gap Identified</span>
                <p className="text-gray-700 leading-relaxed">{gap}</p>
            </div>
            <div className="pl-4 border-l-2 border-green-200">
                <span className="font-bold text-green-700 block mb-1">Relevance to This Project</span>
                <p className="text-gray-700 leading-relaxed">{relevance}</p>
            </div>
        </div>
    </div>
);

const GapCard: React.FC<{ number: number; title: string; subtitle: string; description: string; isHighlighted?: boolean }> = ({ number, title, subtitle, description, isHighlighted = false }) => (
    <div className={`rounded-xl p-5 shadow-lg transition-all hover:shadow-xl ${isHighlighted ? 'bg-gradient-to-br from-amber-50 to-orange-50 border-2 border-amber-300' : 'bg-white border-2 border-amber-200'}`}>
        <div className="flex items-center gap-2.5 mb-3">
            <div className={`w-10 h-10 rounded-full ${isHighlighted ? 'bg-gradient-to-br from-amber-500 to-orange-500 shadow-lg' : 'bg-amber-100'} flex items-center justify-center`}>
                <span className={`text-lg font-bold ${isHighlighted ? 'text-white' : 'text-amber-700'}`}>{number}</span>
            </div>
            <div>
                <h3 className="font-bold text-gray-900">{title}</h3>
                <p className="text-xs text-gray-600">{subtitle}</p>
            </div>
        </div>
        <p className="text-gray-700 leading-relaxed text-sm">{description}</p>
        {isHighlighted && (
            <div className="mt-3 pt-3 border-t border-amber-200">
                <span className="inline-block px-3 py-1 bg-amber-200 rounded-full text-xs font-bold text-amber-800 uppercase tracking-wide">
                    Most Critical Gap
                </span>
            </div>
        )}
    </div>
);

const ContributionCard: React.FC<{ title: string; subtitle: string; description: string; gradient: string; icon: string }> = ({ title, subtitle, description, gradient, icon }) => (
    <div className="bg-white rounded-xl p-5 border-2 border-gray-200 shadow-lg hover:shadow-2xl transition-all hover:border-primary-200">
        <div className={`w-12 h-12 bg-gradient-to-br ${gradient} rounded-xl flex items-center justify-center text-2xl mb-3 shadow-lg`}>
            {icon}
        </div>
        <h3 className="font-bold text-gray-900 text-lg mb-1.5">{title}</h3>
        <p className="text-xs text-primary-600 font-semibold mb-3">{subtitle}</p>
        <p className="text-gray-700 leading-relaxed text-sm">{description}</p>
    </div>
);

const ResearchQuestion: React.FC<{ number: string; question: string }> = ({ number, question }) => (
    <div className="bg-gradient-to-r from-gray-50 to-blue-50 border-l-4 border-primary-500 p-4 rounded-xl shadow-md hover:shadow-lg transition-shadow">
        <div className="flex items-start gap-3">
            <div className="w-10 h-10 bg-primary-500 rounded-full flex items-center justify-center flex-shrink-0 shadow-lg">
                <span className="font-bold text-white text-sm">{number.replace('RQ', '')}</span>
            </div>
            <div>
                <span className="font-bold text-primary-700 block mb-1.5">{number}</span>
                <p className="text-gray-800 leading-relaxed text-sm">{question}</p>
            </div>
        </div>
    </div>
);
