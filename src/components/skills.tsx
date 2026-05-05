import Tilt from 'react-parallax-tilt';
import { motion, Variants } from 'framer-motion';
import Image from 'next/image';
import { useRef } from 'react';
import { services } from '../constants';
import { fadeIn, textVariant } from '../utils/motion';
import SectionWrapper from '../components/sectionWrapper';

const ServiceCard = ({ index, title, icons }: { index: number; title: string; icons: any }) => (
  <Tilt className='pointer-events-auto w-full' scale={1.05} transitionSpeed={450} tiltMaxAngleX={12} tiltMaxAngleY={12}>
    <motion.div
      variants={fadeIn('right', 'spring', index * 0.5, 0.75) as Variants}
      className='w-full rounded-[20px] bg-gradient-90 from-primary to-accent p-[2px]'
    >
      <h3 className='py-2 text-center font-mono text-[18px] font-bold text-text'>{title}</h3>
      <div className='flex min-h-[260px] flex-col justify-evenly rounded-[20px] bg-background px-6 py-5'>
        {icons.map((icon: any) => (
          <div
            key={icon.name}
            onClick={() => window.open(icon.link, '_blank')}
            onKeyDown={(event) => {
              if (event.key === 'Enter') {
                window.open(icon.link, '_blank');
              }
            }}
            className='flex cursor-pointer items-center transition-transform hover:scale-110 focus:scale-110'
            tabIndex={0}
          >
            <Image className='h-6 w-6' src={icon.icon} alt={icon.name} />
            <span className='ml-2 py-1 font-mono text-sm font-medium text-text'>{icon.name}</span>
          </div>
        ))}
      </div>
    </motion.div>
  </Tilt>
);

const ArrowButton = ({
  direction,
  onClick,
}: {
  direction: 'left' | 'right';
  onClick: () => void;
}) => (
  <button
    type='button'
    aria-label={direction === 'left' ? 'Scroll skills left' : 'Scroll skills right'}
    onClick={onClick}
    className={`pointer-events-auto absolute top-1/2 z-20 hidden h-11 w-11 -translate-y-1/2 items-center justify-center rounded-full border border-accent/40 bg-background/90 text-text shadow-card backdrop-blur transition-transform duration-200 hover:scale-110 hover:text-primary focus:scale-110 focus:text-primary sm:flex ${
      direction === 'left' ? 'left-0' : 'right-0'
    }`}
  >
    <svg viewBox='0 0 320 512' className='h-4 w-4'>
      {direction === 'left' ? (
        <path
          fill='currentColor'
          d='M15.1 239c-9.4 9.4-9.4 24.6 0 33.9l160 160c9.4 9.4 24.6 9.4 33.9 0s9.4-24.6 0-33.9L65.9 256 209 112.9c9.4-9.4 9.4-24.6 0-33.9s-24.6-9.4-33.9 0l-160 160z'
        />
      ) : (
        <path
          fill='currentColor'
          d='M304.9 239c9.4 9.4 9.4 24.6 0 33.9l-160 160c-9.4 9.4-24.6 9.4-33.9 0s-9.4-24.6 0-33.9L254.1 256 111 112.9c-9.4-9.4-9.4-24.6 0-33.9s24.6-9.4 33.9 0l160 160z'
        />
      )}
    </svg>
  </button>
);

const Skills = () => {
  const scrollerRef = useRef<HTMLDivElement>(null);

  const scrollByAmount = (direction: 'left' | 'right') => {
    const el = scrollerRef.current;
    if (!el) return;
    const amount = Math.max(280, Math.floor(el.clientWidth * 0.8));
    el.scrollBy({ left: direction === 'left' ? -amount : amount, behavior: 'smooth' });
  };

  return (
    <>
      <motion.div variants={textVariant() as Variants}>
        <p className='section-subtitle'>My capabilities</p>
        <h2 className='section-title'>Skills.</h2>
      </motion.div>

      <div className='relative mt-10 sm:px-16'>
        <ArrowButton direction='left' onClick={() => scrollByAmount('left')} />
        <ArrowButton direction='right' onClick={() => scrollByAmount('right')} />

        <div
          ref={scrollerRef}
          className='no-scrollbar flex snap-x snap-mandatory gap-6 overflow-x-auto px-1 pb-6 pt-2'
        >
          {services.map((service, index) => (
            <div key={service.title} className='w-[260px] flex-none snap-start'>
              <ServiceCard index={index} {...service} />
            </div>
          ))}
        </div>
      </div>
    </>
  );
};

export default SectionWrapper(Skills, 'skills', { tinted: true });
