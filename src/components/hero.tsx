import React, { useEffect, useRef } from 'react';
import letterBounce, { bounce } from '../utils/letterBounce';
import { motion, Variants } from 'framer-motion';
import fluidHover from '../utils/fluidHover';
import webGLFluidEnhanced from 'webgl-fluid-enhanced';
import { fadeIn, appear } from '../utils/motion';
import HeroImage from './heroImage';
import HeroSocial from './heroSocial';

const Hero = () => {
  const spanRefs = useRef<Array<HTMLSpanElement>>([]);

  useEffect(() => {
    const scrollButton = document.querySelector('.scroll-button') as HTMLElement;
    if (scrollButton) {
      fluidHover(scrollButton);
      scrollButton.addEventListener('click', () => {
        webGLFluidEnhanced.splats();
      });
    }

    const aboutText = document.querySelector('.about-text') as HTMLElement;
    if (aboutText) fluidHover(aboutText);

    return letterBounce('span.bouncer');
  }, []);

  const handleSpanRef = (span: HTMLSpanElement) => {
    if (span && !spanRefs.current.includes(span)) {
      spanRefs.current.push(span);
    }
  };

  const handleAppearComplete = (span: HTMLElement) => {
    bounce(span);
  };

  return (
    <section className='relative z-10 mx-auto flex h-screen-large w-full select-none items-center justify-center'>
      <div className='paddingX pointer-events-auto flex w-full max-w-4xl flex-col items-center gap-8 text-center'>

        {/* Photo */}
        <HeroImage />

        {/* Heading */}
        <h1 className='flex flex-wrap justify-center text-[40px] font-black leading-[1.1] text-text xs:text-[50px] sm:text-[60px] lg:text-[72px]'>
          <div>
            <motion.span
              ref={handleSpanRef}
              className='bouncer inline-block transition-colors hover:text-accent'
              variants={appear(0.1, 0.5, () => handleAppearComplete(spanRefs.current[0])) as Variants}
              initial='hidden'
              animate='show'
            >
              H
            </motion.span>
            <motion.span
              ref={handleSpanRef}
              className='bouncer inline-block transition-colors hover:text-accent'
              variants={appear(0.2, 0.5, () => handleAppearComplete(spanRefs.current[1])) as Variants}
              initial='hidden'
              animate='show'
            >
              i
            </motion.span>
            <motion.span
              ref={handleSpanRef}
              className='bouncer inline-block transition-colors hover:text-accent'
              variants={appear(0.3, 0.5, () => handleAppearComplete(spanRefs.current[2])) as Variants}
              initial='hidden'
              animate='show'
            >
              ,&nbsp;
            </motion.span>
          </div>
          <div>
            <motion.span
              ref={handleSpanRef}
              className='bouncer inline-block transition-colors hover:text-accent'
              variants={appear(0.4, 0.5, () => handleAppearComplete(spanRefs.current[3])) as Variants}
              initial='hidden'
              animate='show'
            >
              I
            </motion.span>
            <motion.span
              ref={handleSpanRef}
              className='bouncer inline-block transition-colors hover:text-accent'
              variants={appear(0.5, 0.5, () => handleAppearComplete(spanRefs.current[4])) as Variants}
              initial='hidden'
              animate='show'
            >
              &apos;
            </motion.span>
            <motion.span
              ref={handleSpanRef}
              className='bouncer inline-block transition-colors hover:text-accent'
              variants={appear(0.6, 0.5, () => handleAppearComplete(spanRefs.current[5])) as Variants}
              initial='hidden'
              animate='show'
            >
              m&nbsp;
            </motion.span>
          </div>
          <div className='flex'>
            <span className='gradient-animation from-primary to-accent bg-big bg-clip-text text-transparent bg-gradient-120'>
              <motion.span
                ref={handleSpanRef}
                className='bouncer gradient-letter inline-block transition-colors hover:text-accent'
                variants={appear(0.5, 0.5, () => handleAppearComplete(spanRefs.current[6])) as Variants}
                initial='hidden'
                animate='show'
              >
                N
              </motion.span>
              <motion.span
                ref={handleSpanRef}
                className='bouncer gradient-letter inline-block transition-colors hover:text-accent'
                variants={appear(0.6, 0.5, () => handleAppearComplete(spanRefs.current[7])) as Variants}
                initial='hidden'
                animate='show'
              >
                i
              </motion.span>
              <motion.span
                ref={handleSpanRef}
                className='bouncer gradient-letter inline-block transition-colors hover:text-accent'
                variants={appear(0.7, 0.5, () => handleAppearComplete(spanRefs.current[8])) as Variants}
                initial='hidden'
                animate='show'
              >
                t
              </motion.span>
              <motion.span
                ref={handleSpanRef}
                className='bouncer gradient-letter inline-block transition-colors hover:text-accent'
                variants={appear(0.8, 0.5, () => handleAppearComplete(spanRefs.current[9])) as Variants}
                initial='hidden'
                animate='show'
              >
                i
              </motion.span>
              <motion.span
                ref={handleSpanRef}
                className='bouncer gradient-letter inline-block transition-colors hover:text-accent'
                variants={appear(0.9, 0.5, () => handleAppearComplete(spanRefs.current[10])) as Variants}
                initial='hidden'
                animate='show'
              >
                n
              </motion.span>
            </span>
            <motion.span
              ref={handleSpanRef}
              className='bouncer inline-block transition-colors hover:text-accent'
              variants={appear(1.9, 0, () => handleAppearComplete(spanRefs.current[13])) as Variants}
              initial='hidden'
              animate='show'
            >
              .
            </motion.span>
          </div>
        </h1>

        {/* Subtitle */}
        <motion.p
          className='about-text max-w-2xl text-[15px] font-medium leading-[26px] text-text xs:text-[17px] xs:leading-[28px] sm:text-[20px] sm:leading-[32px] lg:text-[22px] lg:leading-[36px]'
          variants={fadeIn('', '', 1.5, 1) as Variants}
          initial='hidden'
          animate='show'
        >
          I&apos;m a <span className='from-primary to-accent bg-clip-text text-transparent bg-gradient-120'>Machine Learning Engineer focused on production LLM systems</span> based in Jersey City, NJ —
          building agentic pipelines, RAG systems, and cloud-native NLP with measurable impact.
        </motion.p>

        {/* Social links */}
        <HeroSocial />

      </div>
    </section>
  );
};

export default Hero;
