import { Variants, motion } from 'framer-motion';
import { ComponentType } from 'react';
import { staggerContainer } from '../utils/motion';
import { useEffect } from 'react';
import fluidHover from '../utils/fluidHover';

type WrapperOptions = {
  tinted?: boolean;
};

const SectionWrapper = (Component: ComponentType, idName: string, options: WrapperOptions = {}) => {
  const { tinted = false } = options;

  const WrappedComponent = (props: any) => {
    useEffect(() => {
      const sectionTitles = document.querySelectorAll('.section-title') as NodeListOf<HTMLElement>;
      const sectionTexts = document.querySelectorAll('.section-text') as NodeListOf<HTMLElement>;

      sectionTitles.forEach((sectionTitle) => {
        fluidHover(sectionTitle);
      });

      sectionTexts.forEach((sectionText) => {
        fluidHover(sectionText);
      });
    }, []);
    return (
      <div className={`section-shell relative z-10 w-full ${tinted ? 'tinted' : ''}`}>
        <motion.section
          id={idName}
          variants={staggerContainer() as Variants}
          initial='hidden'
          whileInView='show'
          viewport={{ once: true, amount: 0.15 }}
          className='section-padding mx-auto max-w-7xl scroll-mt-[100px]'
        >
          <Component {...props} />
        </motion.section>
      </div>
    );
  };

  WrappedComponent.displayName = `SectionWrapper(${Component.displayName || Component.name || 'Component'})`;

  return WrappedComponent;
};

export default SectionWrapper;
