import React from 'react';
import { BrowserRouter, Routes, Route, Link, useLocation } from 'react-router-dom';
import { HomePage } from './pages/HomePage';
import { AssessmentPage } from './pages/AssessmentPage';
import { CandidatesPage } from './pages/CandidatesPage';
import { RubricPage } from './pages/RubricPage';
import { MethodologyPage } from './pages/MethodologyPage';
import { ResearchPage } from './pages/ResearchPage';
import { Mic, Users, Home, BookOpen, Activity, FlaskConical } from 'lucide-react';

function App() {
  return (
    <BrowserRouter>
      <div className="min-h-screen flex flex-col">
        <Navigation />
        <main className="flex-1">
          <Routes>
            <Route path="/" element={<HomePage />} />
            <Route path="/assessment" element={<AssessmentPage />} />
            <Route path="/candidates" element={<CandidatesPage />} />
            <Route path="/rubric" element={<RubricPage />} />
            <Route path="/methodology" element={<MethodologyPage />} />
            <Route path="/research" element={<ResearchPage />} />
          </Routes>
        </main>
        <Footer />
      </div>
    </BrowserRouter>
  );
}

const Navigation: React.FC = () => {
  const location = useLocation();

  const isActive = (path: string) => {
    return location.pathname === path;
  };

  return (
    <nav className="bg-white shadow-sm border-b border-gray-200">
      <div className="container mx-auto px-4">
        <div className="flex items-center justify-between h-16">
          {/* Logo */}
          <Link to="/" className="flex items-center space-x-2">
            <div className="w-10 h-10 bg-gradient-to-br from-primary-600 to-violet-600 rounded-lg flex items-center justify-center">
              <Mic className="w-6 h-6 text-white" />
            </div>
            <span className="text-xl font-bold text-gray-900">
              Voice XAI Recruit
            </span>
          </Link>

          {/* Navigation Links */}
          <div className="flex space-x-1">
            <NavLink
              to="/"
              icon={<Home className="w-5 h-5" />}
              label="Home"
              active={isActive('/')}
            />
            <NavLink
              to="/assessment"
              icon={<Mic className="w-5 h-5" />}
              label="Live Analysis"
              active={isActive('/assessment')}
            />
            <NavLink
              to="/candidates"
              icon={<Users className="w-5 h-5" />}
              label="Candidates"
              active={isActive('/candidates')}
            />
            <NavLink
              to="/rubric"
              icon={<BookOpen className="w-5 h-5" />}
              label="Rubric"
              active={isActive('/rubric')}
            />
            <NavLink
              to="/methodology"
              icon={<Activity className="w-5 h-5" />}
              label="How It Works"
              active={isActive('/methodology')}
            />
            <NavLink
              to="/research"
              icon={<FlaskConical className="w-5 h-5" />}
              label="Research"
              active={isActive('/research')}
            />
          </div>
        </div>
      </div>
    </nav>
  );
};

interface NavLinkProps {
  to: string;
  icon: React.ReactNode;
  label: string;
  active: boolean;
}

const NavLink: React.FC<NavLinkProps> = ({ to, icon, label, active }) => (
  <Link
    to={to}
    className={`
      flex items-center space-x-2 px-4 py-2 rounded-lg transition-colors
      ${active
        ? 'bg-primary-50 text-primary-700 font-semibold'
        : 'text-gray-600 hover:bg-gray-50 hover:text-gray-900'
      }
    `}
  >
    {icon}
    <span>{label}</span>
  </Link>
);

const Footer: React.FC = () => (
  <footer className="bg-white border-t border-gray-200 py-8 mt-12">
    <div className="container mx-auto px-4 text-center text-gray-600">
      <p className="font-semibold mb-2">Voice-Based Explainable AI for Recruitment</p>
      <p className="text-sm">Final Year Project | Software Engineering Soft Skills Assessment</p>
      <p className="text-sm mt-2">Using Random Forest + SHAP for Transparent, Fair Hiring</p>
    </div>
  </footer>
);

export default App;
