import { motion, Variants } from 'framer-motion';
import { fadeIn, textVariant } from '../utils/motion';
import SectionWrapper from '../components/sectionWrapper';

const About = () => {
  return (
    <>
      <motion.div variants={textVariant() as Variants}>
        <p className='section-subtitle'>Introduction</p>
        <h2 className='section-title'>Overview.</h2>
      </motion.div>

      <motion.p variants={fadeIn('', '', 0.1, 1) as Variants} className='section-text pointer-events-auto mt-4 max-w-3xl text-[17px] leading-[30px] text-text'>
        Machine Learning Engineer with experience building production-grade NLP and LLM systems across AWS and Azure. I focus on multi-agent architectures,
        retrieval-augmented generation, and reliable model deployment with measurable business impact. My work spans fine-tuning, inference optimization,
        observability, and human-in-the-loop evaluation, with a strong preference for practical, testable AI products that scale.
      </motion.p>
    </>
  );
};

export default SectionWrapper(About, 'about');
